# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import math
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.models.image_restoration_model import ImageRestorationModel 
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.data.degradations_dasr import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, only_generate_gaussian_noise_pt, only_generate_poisson_noise_pt, add_given_gaussian_noise_pt, add_given_poisson_noise_pt,add_gaussian_noise_pt,add_poisson_noise_pt
from basicsr.data.transforms_dasr import paired_random_crop_return_indexes

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
from basicsr.utils import DiffJPEG, USMSharp
#from basicsr.data.degradations import random_add_gaussian_noise_pt
from basicsr.utils.img_process_util import filter2D

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class DASR_bic_Model(ImageRestorationModel):
    def __init__(self, opt):
        super(DASR_bic_Model, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt['queue_size']
        self.resize_mode_list = ['area', 'bilinear', 'bicubic']
        self.opt_train = opt['datasets']['train']
        num_degradation_params = 4 * 2 + 2 # kernel
        num_degradation_params += 4 * 2 # resize
        num_degradation_params += 4 * 2 # noise
        num_degradation_params += 3 + 2 + 2 # jpeg
        self.num_degradation_params = num_degradation_params
        self.road_map = [0,
                         10,
                         10 + 8,
                         10 + 8 + 8,
                         10 + 8 + 8 + 7]
        # [0, 10, 18, 26, 33]
        # define network

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        # training pair pool
        # initializeS
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, 'queue size should be divisible by batch size'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    

    def feed_data(self, data_all):
        if self.is_train:
            # training data synthesis
            self.degradation_degree = random.choices(self.opt['degree_list'], self.opt['degree_prob'])[0]
            data = data_all[self.degradation_degree]
            # data = data_all
            self.gt_L = data['gt_L'].to(self.device)
            self.gt_R = data['gt_R'].to(self.device)
            #self.gt = torch.cat([self.gt_L, self.gt_R], dim=1)
            #print('12',(self.gt_R).shape)
            
            self.gt_L = self.usm_sharpener(self.gt_L)
            self.gt_R = self.usm_sharpener(self.gt_R)

            self.gt_for_cycle = self.gt_R.clone()

            if self.degradation_degree == 'severe_degrade_two_stage':

                self.degradation_params = torch.zeros(self.opt_train['batch_size_per_gpu'], self.num_degradation_params)  # [B, 33]

                self.kernel1 = data['kernel1']['kernel'].to(self.device)
                self.kernel2 = data['kernel2']['kernel'].to(self.device)
                self.sinc_kernel = data['sinc_kernel']['kernel'].to(self.device)

                kernel_size_range1 = [self.opt_train['blur_kernel_size_minimum'], self.opt_train['blur_kernel_size']]
                kernel_size_range2 = [self.opt_train['blur_kernel_size2_minimum'], self.opt_train['blur_kernel_size2']]
                rotation_range = [-math.pi, math.pi]
                omega_c_range = [np.pi / 3, np.pi]
                self.degradation_params[:, self.road_map[0]:self.road_map[0]+1] = (data['kernel1']['kernel_size'].unsqueeze(1) - kernel_size_range1[0]) / (kernel_size_range1[1] - kernel_size_range1[0])
                self.degradation_params[:, self.road_map[0]+4:self.road_map[0]+5] = (data['kernel2']['kernel_size'].unsqueeze(1) - kernel_size_range2[0]) / (kernel_size_range2[1] - kernel_size_range2[0])
                self.degradation_params[:, self.road_map[0]+1:self.road_map[0]+2] = (data['kernel1']['sigma_x'].unsqueeze(1) - self.opt_train['blur_sigma'][0]) / (self.opt_train['blur_sigma'][1] - self.opt_train['blur_sigma'][0])
                self.degradation_params[:, self.road_map[0]+5:self.road_map[0]+6] = (data['kernel2']['sigma_x'].unsqueeze(1) - self.opt_train['blur_sigma2'][0]) / (self.opt_train['blur_sigma2'][1] - self.opt_train['blur_sigma2'][0])
                self.degradation_params[:, self.road_map[0]+2:self.road_map[0]+3] = (data['kernel1']['sigma_y'].unsqueeze(1) - self.opt_train['blur_sigma'][0]) / (self.opt_train['blur_sigma'][1] - self.opt_train['blur_sigma'][0])
                self.degradation_params[:, self.road_map[0]+6:self.road_map[0]+7] = (data['kernel2']['sigma_y'].unsqueeze(1) - self.opt_train['blur_sigma2'][0]) / (self.opt_train['blur_sigma2'][1] - self.opt_train['blur_sigma2'][0])
                self.degradation_params[:, self.road_map[0]+3:self.road_map[0]+4] = (data['kernel1']['rotation'].unsqueeze(1) - rotation_range[0]) / (rotation_range[1] - rotation_range[0])
                self.degradation_params[:, self.road_map[0]+7:self.road_map[0]+8] = (data['kernel2']['rotation'].unsqueeze(1) - rotation_range[0]) / (rotation_range[1] - rotation_range[0])
                self.degradation_params[:, self.road_map[0]+8:self.road_map[0]+9] = (data['sinc_kernel']['kernel_size'].unsqueeze(1) - kernel_size_range1[0]) / (kernel_size_range1[1] - kernel_size_range1[0])
                self.degradation_params[:, self.road_map[0]+9:self.road_map[1]] = (data['sinc_kernel']['omega_c'].unsqueeze(1) - omega_c_range[0]) / (omega_c_range[1] - omega_c_range[0])

                ori_h_L, ori_w_L = self.gt_L.size()[2:4]
                ori_h_R, ori_w_R = self.gt_R.size()[2:4]

                # ----------------------- The first degradation process ----------------------- #
                # blur
                out_L = filter2D(self.gt_L, self.kernel1)
                out_R = filter2D(self.gt_R, self.kernel1)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range'][0], 1)
                else:
                    scale = 1
                mode = random.choice(self.resize_mode_list)
                out_L = F.interpolate(out_L, scale_factor=scale, mode=mode)
                out_R = F.interpolate(out_R, scale_factor=scale, mode=mode)
                
                normalized_scale = (scale - self.opt['resize_range'][0]) / (self.opt['resize_range'][1] - self.opt['resize_range'][0])
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                # self.degradation_params[:, self.road_map[1]:self.road_map[1] + 1] = torch.tensor(normalized_scale).expand(self.gt.size(0), 1)
                # self.degradation_params[:, self.road_map[1] + 1:self.road_map[1] + 4] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                # noise # noise_range: [1, 30] poisson_scale_range: [0.05, 3]
                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    sigma, gray_noise, out_L, self.noise_g_first = random_add_gaussian_noise_pt(
                        out_L, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    out_R = add_gaussian_noise_pt(
                        out_R, sigma=sigma, clip=True, rounds=False, gray_noise=gray_noise)
                   
                    normalized_sigma = (sigma - self.opt['noise_range'][0]) / (self.opt['noise_range'][1] - self.opt['noise_range'][0])
                    # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_sigma.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    # self.noise_p_first = only_generate_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_noise_prob)
                else:
                    scale, gray_noise, out_L, self.noise_p_first = random_add_poisson_noise_pt(
                        out_L, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True, rounds=False)
                    out_R = add_poisson_noise_pt(
                        out_R, scale=scale, clip=True, rounds=False, gray_noise=gray_noise)
                    
                    normalized_scale = (scale - self.opt['poisson_scale_range'][0]) / (self.opt['poisson_scale_range'][1] - self.opt['poisson_scale_range'][0])
                    # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_scale.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                    # self.noise_g_first = only_generate_gaussian_noise_pt(out, sigma_range=self.opt['noise_range'], gray_prob=gray_noise_prob)

                # JPEG compression
                seed = random.randint(1, 1000000)  # 可以选择任何整数作为种子
                torch.manual_seed(seed)
                jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range']) # tensor([61.6463, 94.2723, 37.1205, 34.9564], device='cuda:0')]
                normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range'][0]) / (self.opt['jpeg_range'][1] - self.opt['jpeg_range'][0])
                out_L = torch.clamp(out_L, 0, 1)
                out_L = self.jpeger(out_L, quality=jpeg_p_L)

                torch.manual_seed(seed)
                jpeg_p_R = out_R.new_zeros(out_R.size(0)).uniform_(*self.opt['jpeg_range']) # tensor([61.6463, 94.2723, 37.1205, 34.9564], device='cuda:0')]
                normalized_jpeg_p_R = (jpeg_p_R - self.opt['jpeg_range'][0]) / (self.opt['jpeg_range'][1] - self.opt['jpeg_range'][0])
                out_R = torch.clamp(out_R, 0, 1)
                out_R = self.jpeger(out_R, quality=jpeg_p_R)
                #print('*************************---------------------------')
                # self.degradation_params[:, self.road_map[3]:self.road_map[3]+1] = normalized_jpeg_p.unsqueeze(1)

                # ----------------------- The second degradation process ----------------------- #
                # blur
                if np.random.uniform() < self.opt['second_blur_prob']:
                    out_L = filter2D(out_L, self.kernel2)
                    out_R = filter2D(out_R, self.kernel2)
                    # self.degradation_params[:, self.road_map[1] - 1:self.road_map[1]] = torch.tensor([1]).expand(self.gt.size(0), 1)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range2'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(self.resize_mode_list)
                out_L = F.interpolate(out_L, size=(int(ori_h_L / self.opt['scale'] * scale), int(ori_w_L / self.opt['scale'] * scale)), mode=mode)
                out_R = F.interpolate(out_R, size=(int(ori_h_R / self.opt['scale'] * scale), int(ori_w_R / self.opt['scale'] * scale)), mode=mode)
                
                normalized_scale = (scale - self.opt['resize_range2'][0]) / (self.opt['resize_range2'][1] - self.opt['resize_range2'][0])
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                # self.degradation_params[:, self.road_map[1] + 4:self.road_map[1] + 5] = torch.tensor(normalized_scale).expand(self.gt.size(0), 1)
                # self.degradation_params[:, self.road_map[1] + 5:self.road_map[2]] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                # noise
                gray_noise_prob = self.opt['gray_noise_prob2']
                if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                    sigma, gray_noise, out_L, self.noise_g_second = random_add_gaussian_noise_pt(
                        out_L, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    out_R = add_gaussian_noise_pt(
                        out_R, sigma =sigma , clip=True, rounds=False, gray_noise = gray_noise)
                    
                    normalized_sigma = (sigma - self.opt['noise_range2'][0]) / (self.opt['noise_range2'][1] - self.opt['noise_range2'][0])
                    # self.degradation_params[:, self.road_map[2] + 4:self.road_map[2] + 5] = normalized_sigma.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 5:self.road_map[2] + 6] = gray_noise.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 6:self.road_map[3]] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    # self.noise_p_second = only_generate_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range2'], gray_prob=gray_noise_prob)
                else:
                    scale, gray_noise, out_L, self.noise_p_second = random_add_poisson_noise_pt(
                        out_L,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    out_R = add_poisson_noise_pt(
                        out_R,
                        scale = scale,
                        gray_noise = gray_noise,
                        clip=True,
                        rounds=False)
                    normalized_scale = (scale - self.opt['poisson_scale_range2'][0]) / (self.opt['poisson_scale_range2'][1] - self.opt['poisson_scale_range2'][0])
                    # self.degradation_params[:, self.road_map[2] + 4:self.road_map[2] + 5] = normalized_scale.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 5:self.road_map[2] + 6] = gray_noise.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 6:self.road_map[3]] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                    # self.noise_g_second = only_generate_gaussian_noise_pt(out, sigma_range=self.opt['noise_range2'], gray_prob=gray_noise_prob)

                # JPEG compression + the final sinc filter
                if np.random.uniform() < 0.5:
                    # resize back + the final sinc filter
                    mode = random.choice(self.resize_mode_list)
                    onehot_mode = torch.zeros(len(self.resize_mode_list))
                    for index, mode_current in enumerate(self.resize_mode_list):
                        if mode_current == mode:
                            onehot_mode[index] = 1
                    out_L = F.interpolate(out_L, size=(ori_h_L // self.opt['scale'], ori_w_L // self.opt['scale']), mode=mode)
                    out_L = filter2D(out_L, self.sinc_kernel)
                    
                    out_R = F.interpolate(out_R, size=(ori_h_R // self.opt['scale'], ori_w_R // self.opt['scale']), mode=mode)
                    out_R = filter2D(out_R, self.sinc_kernel)

                    # JPEG compression
                    seed = random.randint(1, 1000000)  # 可以选择任何整数作为种子
                    torch.manual_seed(seed)
                    jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range2'])
                    normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range2'][0]) / (self.opt['jpeg_range2'][1] - self.opt['jpeg_range2'][0])
                    out_L = torch.clamp(out_L, 0, 1)
                    out_L = self.jpeger(out_L, quality=jpeg_p_L)
                    # self.degradation_params[:, self.road_map[3] + 1:self.road_map[3] + 2] = normalized_jpeg_p.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[3] + 2:self.road_map[3] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    # self.degradation_params[:, self.road_map[3] + 4:] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                
                    torch.manual_seed(seed)
                    jpeg_p_R = out_R.new_zeros(out_R.size(0)).uniform_(*self.opt['jpeg_range2'])
                    normalized_jpeg_p_R = (jpeg_p_R - self.opt['jpeg_range2'][0]) / (self.opt['jpeg_range2'][1] - self.opt['jpeg_range2'][0])
                    out_R = torch.clamp(out_R, 0, 1)
                    out_R = self.jpeger(out_R, quality=jpeg_p_R)
                    # self.degradation_params[:, self.road_map[3] + 1:self.road_map[3] + 2] = normalized_jpeg_p.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[3] + 2:self.road_map[3] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    # self.degradation_params[:, self.road_map[3] + 4:] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                
                else:
                    # JPEG compression
                    seed = random.randint(1, 1000000)  # 可以选择任何整数作为种子
                    torch.manual_seed(seed)
                    jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range2'])
                    normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range2'][0]) / (self.opt['jpeg_range2'][1] - self.opt['jpeg_range2'][0])
                    out_L = torch.clamp(out_L, 0, 1)
                    out_L = self.jpeger(out_L, quality=jpeg_p_L)

                    torch.manual_seed(seed)
                    jpeg_p_R = out_R.new_zeros(out_R.size(0)).uniform_(*self.opt['jpeg_range2'])
                    normalized_jpeg_p_R = (jpeg_p_R - self.opt['jpeg_range2'][0]) / (self.opt['jpeg_range2'][1] - self.opt['jpeg_range2'][0])
                    out_R = torch.clamp(out_R, 0, 1)
                    out_R = self.jpeger(out_R, quality=jpeg_p_R)
                    # resize back + the final sinc filter
                    mode = random.choice(self.resize_mode_list)
                    onehot_mode = torch.zeros(len(self.resize_mode_list))
                    for index, mode_current in enumerate(self.resize_mode_list):
                        if mode_current == mode:
                            onehot_mode[index] = 1
                    out_L = F.interpolate(out_L, size=(ori_h_L // self.opt['scale'], ori_w_L // self.opt['scale']), mode=mode)
                    out_L = filter2D(out_L, self.sinc_kernel)
                    out_R = F.interpolate(out_R, size=(ori_h_R // self.opt['scale'], ori_w_R // self.opt['scale']), mode=mode)
                    out_R = filter2D(out_R, self.sinc_kernel)
                    # self.degradation_params[:, self.road_map[3] + 1:self.road_map[3] + 2] = normalized_jpeg_p.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[3] + 2:self.road_map[3] + 4] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                    # self.degradation_params[:, self.road_map[3] + 4:] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                # print(self.degradation_params)

                # self.degradation_params = self.degradation_params.to(self.device)

                # clamp and round
                self.lq_L = torch.clamp((out_L * 255.0).round(), 0, 255) / 255.
                self.lq_R = torch.clamp((out_R * 255.0).round(), 0, 255) / 255.

                # random crop
                gt_size = self.opt['gt_size']
                self.gt_L, self.lq_L, self.top, self.left = paired_random_crop_return_indexes(self.gt_L, self.lq_L, gt_size, self.opt['scale'])
                self.gt_R, self.lq_R, self.top, self.left = paired_random_crop_return_indexes(self.gt_R, self.lq_R, gt_size, self.opt['scale'])

                self.gt = torch.cat([self.gt_L, self.gt_R], dim=1)
                self.lq = torch.cat([self.lq_L, self.lq_R], dim=1)
            
            elif self.degradation_degree == 'standard_degrade_one_stage':
                #print('standard*************************---------------------------')

                self.degradation_params = torch.zeros(self.opt_train['batch_size_per_gpu'], self.num_degradation_params)  # [B, 33]

                self.kernel1 = data['kernel1']['kernel'].to(self.device)

                kernel_size_range1 = [self.opt_train['blur_kernel_size_minimum_standard1'], self.opt_train['blur_kernel_size_standard1']]
                rotation_range = [-math.pi, math.pi]
                self.degradation_params[:, self.road_map[0]:self.road_map[0]+1] = (data['kernel1']['kernel_size'].unsqueeze(1) - kernel_size_range1[0]) / (kernel_size_range1[1] - kernel_size_range1[0])
                self.degradation_params[:, self.road_map[0]+1:self.road_map[0]+2] = (data['kernel1']['sigma_x'].unsqueeze(1) - self.opt_train['blur_sigma_standard1'][0]) / (self.opt_train['blur_sigma_standard1'][1] - self.opt_train['blur_sigma_standard1'][0])
                self.degradation_params[:, self.road_map[0]+2:self.road_map[0]+3] = (data['kernel1']['sigma_y'].unsqueeze(1) - self.opt_train['blur_sigma_standard1'][0]) / (self.opt_train['blur_sigma_standard1'][1] - self.opt_train['blur_sigma_standard1'][0])
                self.degradation_params[:, self.road_map[0]+3:self.road_map[0]+4] = (data['kernel1']['rotation'].unsqueeze(1) - rotation_range[0]) / (rotation_range[1] - rotation_range[0])

                ori_h_L, ori_w_L = self.gt_L.size()[2:4]
                ori_h_R, ori_w_R = self.gt_R.size()[2:4]

                # blur
                out_L = filter2D(self.gt_L, self.kernel1)
                out_R = filter2D(self.gt_R, self.kernel1)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob_standard1'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range_standard1'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range_standard1'][0], 1)
                else:
                    scale = 1
                mode = random.choice(self.resize_mode_list)
                out_L = F.interpolate(out_L, scale_factor=scale, mode=mode)
                out_R = F.interpolate(out_R, scale_factor=scale, mode=mode)
                normalized_scale = (scale - self.opt['resize_range_standard1'][0]) / (self.opt['resize_range_standard1'][1] - self.opt['resize_range_standard1'][0])
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                # self.degradation_params[:, self.road_map[1]:self.road_map[1] + 1] = torch.tensor(normalized_scale).expand(self.gt.size(0), 1)
                # self.degradation_params[:, self.road_map[1] + 1:self.road_map[1] + 4] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                # noise # noise_range: [1, 30] poisson_scale_range: [0.05, 3]
                gray_noise_prob = self.opt['gray_noise_prob_standard1']
                if np.random.uniform() < self.opt['gaussian_noise_prob_standard1']:
                    sigma, gray_noise, out_L, self.noise_g_first = random_add_gaussian_noise_pt(
                        out_L, sigma_range=self.opt['noise_range_standard1'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    out_R = add_gaussian_noise_pt(
                        out_R, sigma = sigma, clip=True, rounds=False, gray_noise=gray_noise)
                    normalized_sigma = (sigma - self.opt['noise_range_standard1'][0]) / (self.opt['noise_range_standard1'][1] - self.opt['noise_range_standard1'][0])
                    # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_sigma.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    # self.noise_p_first = only_generate_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range_standard1'], gray_prob=gray_noise_prob)
                else:
                    scale, gray_noise, out_L, self.noise_p_first = random_add_poisson_noise_pt(
                        out_L, scale_range=self.opt['poisson_scale_range_standard1'], gray_prob=gray_noise_prob, clip=True, rounds=False)
                    out_R = add_poisson_noise_pt(
                        out_R, scale = scale, gray_noise=gray_noise, clip=True, rounds=False)
           
                    normalized_scale = (scale - self.opt['poisson_scale_range_standard1'][0]) / (self.opt['poisson_scale_range_standard1'][1] - self.opt['poisson_scale_range_standard1'][0])
                    # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_scale.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                    # self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                    # self.noise_g_first = only_generate_gaussian_noise_pt(out, sigma_range=self.opt['noise_range_standard1'], gray_prob=gray_noise_prob)

                # JPEG compression
                seed = random.randint(1, 1000000)  # 可以选择任何整数作为种子
                torch.manual_seed(seed)
                jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range_standard1']) # tensor([61.6463, 94.2723, 37.1205, 34.9564], device='cuda:0')]
                normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range_standard1'][0]) / (self.opt['jpeg_range_standard1'][1] - self.opt['jpeg_range_standard1'][0])
                out_L = torch.clamp(out_L, 0, 1)
                out_L = self.jpeger(out_L, quality=jpeg_p_L)
                # self.degradation_params[:, self.road_map[3]:self.road_map[3]+1] = normalized_jpeg_p.unsqueeze(1)

                torch.manual_seed(seed)
                jpeg_p_R = out_R.new_zeros(out_R.size(0)).uniform_(*self.opt['jpeg_range_standard1']) # tensor([61.6463, 94.2723, 37.1205, 34.9564], device='cuda:0')]
                normalized_jpeg_p_R = (jpeg_p_R - self.opt['jpeg_range_standard1'][0]) / (self.opt['jpeg_range_standard1'][1] - self.opt['jpeg_range_standard1'][0])
                out_R = torch.clamp(out_R, 0, 1)
                out_R = self.jpeger(out_R, quality=jpeg_p_R)
                # self.degradation_params[:, self.road_map[3]:self.road_map[3]+1] = normalized_jpeg_p.unsqueeze(1)

                # resize back
                mode = random.choice(self.resize_mode_list)
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                out_L = F.interpolate(out_L, size=(ori_h_L // self.opt['scale'], ori_w_L // self.opt['scale']), mode=mode)
                out_R = F.interpolate(out_R, size=(ori_h_R // self.opt['scale'], ori_w_R // self.opt['scale']), mode=mode)
                # self.degradation_params[:, self.road_map[3] + 4:] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))

                # self.degradation_params = self.degradation_params.to(self.device)

                # clamp and round
                self.lq_L = torch.clamp((out_L * 255.0).round(), 0, 255) / 255.
                self.lq_R = torch.clamp((out_R * 255.0).round(), 0, 255) / 255.

                # random crop
                gt_size = self.opt['gt_size']
                self.gt_L, self.lq_L, self.top, self.left = paired_random_crop_return_indexes(self.gt_L, self.lq_L, gt_size, self.opt['scale'])
                self.gt_R, self.lq_R, self.top, self.left = paired_random_crop_return_indexes(self.gt_R, self.lq_R, gt_size, self.opt['scale'])

                self.gt = torch.cat([self.gt_L, self.gt_R], dim=1)
                self.lq = torch.cat([self.lq_L, self.lq_R], dim=1)

            elif self.degradation_degree == 'weak_degrade_one_stage':
                #print('*************************---------------------------')

                self.degradation_params = torch.zeros(self.opt_train['batch_size_per_gpu'], self.num_degradation_params)  # [B, 33]

                self.kernel1 = data['kernel1']['kernel'].to(self.device)

                kernel_size_range1 = [self.opt_train['blur_kernel_size_minimum_weak1'], self.opt_train['blur_kernel_size_weak1']]
                rotation_range = [-math.pi, math.pi]
                self.degradation_params[:, self.road_map[0]:self.road_map[0]+1] = (data['kernel1']['kernel_size'].unsqueeze(1) - kernel_size_range1[0]) / (kernel_size_range1[1] - kernel_size_range1[0])
                self.degradation_params[:, self.road_map[0]+1:self.road_map[0]+2] = (data['kernel1']['sigma_x'].unsqueeze(1) - self.opt_train['blur_sigma_weak1'][0]) / (self.opt_train['blur_sigma_weak1'][1] - self.opt_train['blur_sigma_weak1'][0])
                self.degradation_params[:, self.road_map[0]+2:self.road_map[0]+3] = (data['kernel1']['sigma_y'].unsqueeze(1) - self.opt_train['blur_sigma_weak1'][0]) / (self.opt_train['blur_sigma_weak1'][1] - self.opt_train['blur_sigma_weak1'][0])
                self.degradation_params[:, self.road_map[0]+3:self.road_map[0]+4] = (data['kernel1']['rotation'].unsqueeze(1) - rotation_range[0]) / (rotation_range[1] - rotation_range[0])

                ori_h_L, ori_w_L = self.gt_L.size()[2:4]
                ori_h_R, ori_w_R = self.gt_R.size()[2:4]

                # blur
                out_L = filter2D(self.gt_L, self.kernel1)
                out_R = filter2D(self.gt_R, self.kernel1)
                # random resize
                updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob_weak1'])[0]
                if updown_type == 'up':
                    scale = np.random.uniform(1, self.opt['resize_range_weak1'][1])
                elif updown_type == 'down':
                    scale = np.random.uniform(self.opt['resize_range_weak1'][0], 1)
                else:
                    scale = 1
                mode = random.choice(self.resize_mode_list)
                out_L = F.interpolate(out_L, scale_factor=scale, mode=mode)
                out_R = F.interpolate(out_R, scale_factor=scale, mode=mode)
                normalized_scale = (scale - self.opt['resize_range_weak1'][0]) / (self.opt['resize_range_weak1'][1] - self.opt['resize_range_weak1'][0])
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                # self.degradation_params[:, self.road_map[1]:self.road_map[1] + 1] = torch.tensor(normalized_scale).expand(self.gt.size(0), 1)
                # self.degradation_params[:, self.road_map[1] + 1:self.road_map[1] + 4] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
                # noise # noise_range: [1, 30] poisson_scale_range: [0.05, 3]
                gray_noise_prob = self.opt['gray_noise_prob_weak1']
                if np.random.uniform() < self.opt['gaussian_noise_prob_weak1']:
                    sigma, gray_noise, out_L, self.noise_g_first = random_add_gaussian_noise_pt(
                        out_L, sigma_range=self.opt['noise_range_weak1'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                    out_R = add_gaussian_noise_pt(
                        out_R, sigma = sigma, clip=True, rounds=False, gray_noise=gray_noise)
        
                #     print('2',sigma.shape, gray_noise,out.shape)
                    normalized_sigma = (sigma - self.opt['noise_range_weak1'][0]) / (self.opt['noise_range_weak1'][1] - self.opt['noise_range_weak1'][0])
                #     # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_sigma.unsqueeze(1)
                #     # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                #     self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                    # self.noise_p_first = only_generate_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range_weak1'], gray_prob=gray_noise_prob)
                else:
                    scale, gray_noise, out_L, self.noise_p_first = random_add_poisson_noise_pt(
                        out_L, scale_range=self.opt['poisson_scale_range_weak1'], gray_prob=gray_noise_prob, clip=True, rounds=False)
                    out_R = add_poisson_noise_pt(
                        out_R, scale = scale, gray_noise=gray_noise, clip=True, rounds=False)
                    
                    normalized_scale = (scale - self.opt['poisson_scale_range_weak1'][0]) / (self.opt['poisson_scale_range_weak1'][1] - self.opt['poisson_scale_range_weak1'][0])
                #     self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_scale.unsqueeze(1)
                #     self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                #     self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                 #   self.noise_g_first = only_generate_gaussian_noise_pt(out, sigma_range=self.opt['noise_range_weak1'], gray_prob=gray_noise_prob)

                # JPEG compression
                seed = random.randint(1, 1000000)  # 可以选择任何整数作为种子
                torch.manual_seed(seed)
                jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range_weak1'])
                normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range_weak1'][0]) / (self.opt['jpeg_range_weak1'][1] - self.opt['jpeg_range_weak1'][0])
                out_L = torch.clamp(out_L, 0, 1)
                out_L = self.jpeger(out_L, quality=jpeg_p_L)

                torch.manual_seed(seed)
                jpeg_p_R = out_R.new_zeros(out_R.size(0)).uniform_(*self.opt['jpeg_range_weak1'])
                normalized_jpeg_p = (jpeg_p_R - self.opt['jpeg_range_weak1'][0]) / (self.opt['jpeg_range_weak1'][1] - self.opt['jpeg_range_weak1'][0])
                out_R = torch.clamp(out_R, 0, 1)
                out_R = self.jpeger(out_R, quality=jpeg_p_R)
                #self.degradation_params[:, self.road_map[3]:self.road_map[3]+1] = normalized_jpeg_p.unsqueeze(1)

                # resize back
                mode = random.choice(self.resize_mode_list)
                onehot_mode = torch.zeros(len(self.resize_mode_list))
                for index, mode_current in enumerate(self.resize_mode_list):
                    if mode_current == mode:
                        onehot_mode[index] = 1
                out_L = F.interpolate(out_L, size=(ori_h_L // self.opt['scale'], ori_w_L // self.opt['scale']), mode=mode)
                out_R = F.interpolate(out_R, size=(ori_h_R // self.opt['scale'], ori_w_R // self.opt['scale']), mode=mode)
                
                #self.degradation_params[:, self.road_map[3] + 4:] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))

                #self.degradation_params = self.degradation_params.to(self.device)

                # clamp and round
                self.lq_L = torch.clamp((out_L * 255.0).round(), 0, 255) / 255.
                self.lq_R = torch.clamp((out_R * 255.0).round(), 0, 255) / 255.

                # random crop
                gt_size = self.opt['gt_size']
                self.gt_L, self.lq_L, self.top, self.left = paired_random_crop_return_indexes(self.gt_L, self.lq_L, gt_size, self.opt['scale'])
                self.gt_R, self.lq_R, self.top, self.left = paired_random_crop_return_indexes(self.gt_R, self.lq_R, gt_size, self.opt['scale'])

                self.gt = torch.cat([self.gt_L, self.gt_R], dim=1)
                self.lq = torch.cat([self.lq_L, self.lq_R], dim=1)
            elif self.degradation_degree == 'bic':
                #print('*************************---------------------------')

                ori_h_L, ori_w_L = self.gt_L.size()[2:4]
                ori_h_R, ori_w_R = self.gt_R.size()[2:4]

                # resize back
                mode = self.resize_mode_list[2]
                
                out_L = F.interpolate(self.gt_L, size=(ori_h_L // self.opt['scale'], ori_w_L // self.opt['scale']), mode=mode)
                out_R = F.interpolate(self.gt_R, size=(ori_h_R // self.opt['scale'], ori_w_R // self.opt['scale']), mode=mode)
                
                # clamp and round
                self.lq_L = torch.clamp((out_L * 255.0).round(), 0, 255) / 255.
                self.lq_R = torch.clamp((out_R * 255.0).round(), 0, 255) / 255.

                # random crop
                gt_size = self.opt['gt_size']
                self.gt_L, self.lq_L, self.top, self.left = paired_random_crop_return_indexes(self.gt_L, self.lq_L, gt_size, self.opt['scale'])
                self.gt_R, self.lq_R, self.top, self.left = paired_random_crop_return_indexes(self.gt_R, self.lq_R, gt_size, self.opt['scale'])

                self.gt = torch.cat([self.gt_L, self.gt_R], dim=1)
                self.lq = torch.cat([self.lq_L, self.lq_R], dim=1)
            
            
            else:
                print('Degree Mode Mismatch.')

            # print(self.degradation_params)

            self._dequeue_and_enqueue()
        else:
            data = data_all
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(ImageRestorationModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True



class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_