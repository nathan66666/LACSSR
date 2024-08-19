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


class DASRModel(ImageRestorationModel):
    def __init__(self, opt):
        super(DASRModel, self).__init__(opt)
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




class DASR_val_Model(BaseModel):
    def __init__(self, opt):
        super(DASR_val_Model, self).__init__(opt)
        #define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', False), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        #self.queue_size = opt['queue_size']
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

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
                #             optim_params_lowlr.append(v)
                #         else:
                optim_params.append(v)
                # if "layers_f" in k:
                #     optim_params.append(v)
                # else:
                #     continue
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
    

    def feed_data(self, data_all):

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
                sigma, gray_noise, out_R, self.noise_g_first = random_add_gaussian_noise_pt(
                    out_R, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                
                normalized_sigma = (sigma - self.opt['noise_range'][0]) / (self.opt['noise_range'][1] - self.opt['noise_range'][0])
                # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_sigma.unsqueeze(1)
                # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                # self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                # self.noise_p_first = only_generate_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_noise_prob)
            else:
                scale, gray_noise, out_L, self.noise_p_first = random_add_poisson_noise_pt(
                    out_L, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True, rounds=False)
                scale, gray_noise, out_R, self.noise_p_first = random_add_poisson_noise_pt(
                    out_R, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True, rounds=False)
                
                normalized_scale = (scale - self.opt['poisson_scale_range'][0]) / (self.opt['poisson_scale_range'][1] - self.opt['poisson_scale_range'][0])
                # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_scale.unsqueeze(1)
                # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                # self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                # self.noise_g_first = only_generate_gaussian_noise_pt(out, sigma_range=self.opt['noise_range'], gray_prob=gray_noise_prob)

            # JPEG compression
            jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range']) # tensor([61.6463, 94.2723, 37.1205, 34.9564], device='cuda:0')]
            normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range'][0]) / (self.opt['jpeg_range'][1] - self.opt['jpeg_range'][0])
            out_L = torch.clamp(out_L, 0, 1)
            out_L = self.jpeger(out_L, quality=jpeg_p_L)

            jpeg_p_R = out_R.new_zeros(out_R.size(0)).uniform_(*self.opt['jpeg_range']) # tensor([61.6463, 94.2723, 37.1205, 34.9564], device='cuda:0')]
            normalized_jpeg_p_R = (jpeg_p_R - self.opt['jpeg_range'][0]) / (self.opt['jpeg_range'][1] - self.opt['jpeg_range'][0])
            out_R = torch.clamp(out_R, 0, 1)
            out_R = self.jpeger(out_R, quality=jpeg_p_R)
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
                sigma, gray_noise, out_R, self.noise_g_second = random_add_gaussian_noise_pt(
                    out_R, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                
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
                scale, gray_noise, out_R, self.noise_p_second = random_add_poisson_noise_pt(
                    out_R,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
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
                jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range2'])
                normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range2'][0]) / (self.opt['jpeg_range2'][1] - self.opt['jpeg_range2'][0])
                out_L = torch.clamp(out_L, 0, 1)
                out_L = self.jpeger(out_L, quality=jpeg_p_L)
                # self.degradation_params[:, self.road_map[3] + 1:self.road_map[3] + 2] = normalized_jpeg_p.unsqueeze(1)
                # self.degradation_params[:, self.road_map[3] + 2:self.road_map[3] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                # self.degradation_params[:, self.road_map[3] + 4:] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
            
                jpeg_p_R = out_R.new_zeros(out_R.size(0)).uniform_(*self.opt['jpeg_range2'])
                normalized_jpeg_p_R = (jpeg_p_R - self.opt['jpeg_range2'][0]) / (self.opt['jpeg_range2'][1] - self.opt['jpeg_range2'][0])
                out_R = torch.clamp(out_R, 0, 1)
                out_R = self.jpeger(out_R, quality=jpeg_p_R)
                # self.degradation_params[:, self.road_map[3] + 1:self.road_map[3] + 2] = normalized_jpeg_p.unsqueeze(1)
                # self.degradation_params[:, self.road_map[3] + 2:self.road_map[3] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                # self.degradation_params[:, self.road_map[3] + 4:] = onehot_mode.expand(self.gt.size(0), len(self.resize_mode_list))
            
            else:
                # JPEG compression
                jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range2'])
                normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range2'][0]) / (self.opt['jpeg_range2'][1] - self.opt['jpeg_range2'][0])
                out_L = torch.clamp(out_L, 0, 1)
                out_L = self.jpeger(out_L, quality=jpeg_p_L)

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
                sigma, gray_noise, out_R, self.noise_g_first = random_add_gaussian_noise_pt(
                    out_R, sigma_range=self.opt['noise_range_standard1'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                normalized_sigma = (sigma - self.opt['noise_range_standard1'][0]) / (self.opt['noise_range_standard1'][1] - self.opt['noise_range_standard1'][0])
                # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_sigma.unsqueeze(1)
                # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                # self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                # self.noise_p_first = only_generate_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range_standard1'], gray_prob=gray_noise_prob)
            else:
                scale, gray_noise, out_L, self.noise_p_first = random_add_poisson_noise_pt(
                    out_L, scale_range=self.opt['poisson_scale_range_standard1'], gray_prob=gray_noise_prob, clip=True, rounds=False)
                scale, gray_noise, out_R, self.noise_p_first = random_add_poisson_noise_pt(
                    out_R, scale_range=self.opt['poisson_scale_range_standard1'], gray_prob=gray_noise_prob, clip=True, rounds=False)
        
                normalized_scale = (scale - self.opt['poisson_scale_range_standard1'][0]) / (self.opt['poisson_scale_range_standard1'][1] - self.opt['poisson_scale_range_standard1'][0])
                # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_scale.unsqueeze(1)
                # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
                # self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                # self.noise_g_first = only_generate_gaussian_noise_pt(out, sigma_range=self.opt['noise_range_standard1'], gray_prob=gray_noise_prob)

            # JPEG compression
            jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range_standard1']) # tensor([61.6463, 94.2723, 37.1205, 34.9564], device='cuda:0')]
            normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range_standard1'][0]) / (self.opt['jpeg_range_standard1'][1] - self.opt['jpeg_range_standard1'][0])
            out_L = torch.clamp(out_L, 0, 1)
            out_L = self.jpeger(out_L, quality=jpeg_p_L)
            # self.degradation_params[:, self.road_map[3]:self.road_map[3]+1] = normalized_jpeg_p.unsqueeze(1)

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

            if self.opt['val']:
                from PIL import Image
                import datetime
                
                outputRs = self.lq_L.permute(0,2,3,1)
                k = outputRs.cpu().detach().numpy()
                for i in range(4):
                    res = k[i] #得到batch中其中一步的图片
                    image = Image.fromarray(np.uint8(res)*255).convert('RGB')
                    #image.show()
                    #通过时间命名存储结果
                    timestamp = datetime.datetime.now().strftime("%M-%S")
                    savepath = '/lxy/'+timestamp + '_lq.jpg'
                    image.save(savepath)

            self.gt = torch.cat([self.gt_L, self.gt_R], dim=1)
            self.lq = torch.cat([self.lq_L, self.lq_R], dim=1)

        elif self.degradation_degree == 'weak_degrade_one_stage':

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
                sigma, gray_noise, out_R, self.noise_g_first = random_add_gaussian_noise_pt(
                    out_L, sigma_range=self.opt['noise_range_weak1'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    
            #     print('2',sigma.shape, gray_noise,out.shape)
                normalized_sigma = (sigma - self.opt['noise_range_weak1'][0]) / (self.opt['noise_range_weak1'][1] - self.opt['noise_range_weak1'][0])
            #     # self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_sigma.unsqueeze(1)
            #     # self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
            #     self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([1, 0]).expand(self.gt.size(0), 2)
                # self.noise_p_first = only_generate_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range_weak1'], gray_prob=gray_noise_prob)
            else:
                scale, gray_noise, out_L, self.noise_p_first = random_add_poisson_noise_pt(
                    out_L, scale_range=self.opt['poisson_scale_range_weak1'], gray_prob=gray_noise_prob, clip=True, rounds=False)
                scale, gray_noise, out_R, self.noise_p_first = random_add_poisson_noise_pt(
                    out_R, scale_range=self.opt['poisson_scale_range_weak1'], gray_prob=gray_noise_prob, clip=True, rounds=False)
                
                normalized_scale = (scale - self.opt['poisson_scale_range_weak1'][0]) / (self.opt['poisson_scale_range_weak1'][1] - self.opt['poisson_scale_range_weak1'][0])
            #     self.degradation_params[:, self.road_map[2]:self.road_map[2] + 1] = normalized_scale.unsqueeze(1)
            #     self.degradation_params[:, self.road_map[2] + 1:self.road_map[2] + 2] = gray_noise.unsqueeze(1)
            #     self.degradation_params[:, self.road_map[2] + 2:self.road_map[2] + 4] = torch.tensor([0, 1]).expand(self.gt.size(0), 2)
                #   self.noise_g_first = only_generate_gaussian_noise_pt(out, sigma_range=self.opt['noise_range_weak1'], gray_prob=gray_noise_prob)

            # JPEG compression
            jpeg_p_L = out_L.new_zeros(out_L.size(0)).uniform_(*self.opt['jpeg_range_weak1'])
            normalized_jpeg_p_L = (jpeg_p_L - self.opt['jpeg_range_weak1'][0]) / (self.opt['jpeg_range_weak1'][1] - self.opt['jpeg_range_weak1'][0])
            out_L = torch.clamp(out_L, 0, 1)
            out_L = self.jpeger(out_L, quality=jpeg_p_L)

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

            print(type(self.lq_R),(self.lq_R).shape)
            self.gt = torch.cat([self.gt_L, self.gt_R], dim=1)
            self.lq = torch.cat([self.lq_L, self.lq_R], dim=1)
        else:
            print('Degree Mode Mismatch.')

        # print(self.degradation_params)


    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            #
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.use_chop = self.opt['val']['use_chop'] if 'use_chop' in self.opt['val'] else False
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i1 = 0
            while i1 < n:
                j1 = i1 + m
                if j1 >= n:
                    j1 = n

                if not self.use_chop:
                    if self.opt['network_g']['type'] == 'NAFSSR' or self.opt['network_g']['type'] == 'RRDBNet' or self.opt['network_g']['type'] == 'MSRResNet_prior':
                        img = self.lq  # img
                        pred = self.net_g(img)
                    else:
                        if self.opt['network_g']['type'] == 'NAFSSR' or self.opt['network_g']['type'] == 'RRDBNet' or self.opt['network_g']['type'] == 'MSRResNet_prior': 
                            patch_size1 = max(self.opt['network_g']['split_size_0'])
                            patch_size2 = max(self.opt['network_g']['split_size_1'])
                            patch_size = max(patch_size1, patch_size2)
                            scale = 4

                            mod_pad_h, mod_pad_w = 0, 0
                            _, _, h, w = self.lq.size()
                            if h % patch_size != 0:
                                mod_pad_h = patch_size - h % patch_size
                            if w % patch_size != 0:
                                mod_pad_w = patch_size - w % patch_size

                        if self.opt['network_g']['type'] == 'SWIN2SR':
                            window_size = self.opt['network_g']['window_size']
                            scale = self.opt.get('scale', 1)
                            mod_pad_h, mod_pad_w = 0, 0
                            _, _, h, w = self.lq.size()
                            if h % window_size != 0:
                                mod_pad_h = window_size - h % window_size
                            if w % window_size != 0:
                                mod_pad_w = window_size - w % window_size
                

                        img = self.lq  # img
                        #print(img.shape)

                        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + mod_pad_h, :]
                        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]
                        #print(img.shape)

                        pred = self.net_g(img)  # forward

                        _, _, h, w = pred.size()
                        pred = pred[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
                else:
                    if self.opt['network_g']['type'] == 'NAFSSR':
                        img = self.lq  # img
                        pred = self.net_g(img)
                        _, C, h, w = self.lq.size()
                        split_token_h = h // 30 + 1  # number of horizontal cut sections
                        split_token_w = w // 90 + 1  # number of vertical cut sections

                        _, _, H, W = img.size()
                        split_h = H // split_token_h  # height of each partition
                        split_w = W // split_token_w  # width of each partition

                        # overlapping
                        shave_h = 16
                        shave_w = 16
                        scale = self.opt.get('scale', 1)
                        ral = H // split_h
                        row = W // split_w
                        slices = []  # list of partition borders
                        for i in range(ral):
                            for j in range(row):
                                if i == 0 and i == ral - 1:
                                    top = slice(i * split_h, (i + 1) * split_h)
                                elif i == 0:
                                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                                elif i == ral - 1:
                                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                                else:
                                    top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)
                                if j == 0 and j == row - 1:
                                    left = slice(j * split_w, (j + 1) * split_w)
                                elif j == 0:
                                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                                elif j == row - 1:
                                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                                else:
                                    left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)
                                temp = (top, left)
                                slices.append(temp)
                        img_chops = []  # list of partitions
                        for temp in slices:
                            top, left = temp
                            img_chops.append(img[..., top, left])
                        if hasattr(self, 'net_g_ema'):
                            self.net_g_ema.eval()
                            with torch.no_grad():
                                outputs = []
                                for chop in img_chops:
                                    out = self.net_g_ema(chop)  # image processing of each partition
                                    outputs.append(out)
                                _img = torch.zeros(1, C, H * scale, W * scale)
                                # merge
                                for i in range(ral):
                                    for j in range(row):
                                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                                        if i == 0:
                                            _top = slice(0, split_h * scale)
                                        else:
                                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                                        if j == 0:
                                            _left = slice(0, split_w * scale)
                                        else:
                                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                                pred = _img
                        else:
                            self.net_g.eval()
                            with torch.no_grad():
                                outputs = []
                                for chop in img_chops:
                                    out = self.net_g(chop)  # image processing of each partition
                                    outputs.append(out)
                                _img = torch.zeros(1, C, H * scale, W * scale)
                                # merge
                                for i in range(ral):
                                    for j in range(row):
                                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                                        if i == 0:
                                            _top = slice(0, split_h * scale)
                                        else:
                                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                                        if j == 0:
                                            _left = slice(0, split_w * scale)
                                        else:
                                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                                pred = _img
                            self.net_g.train()
                        _, _, h, w = pred.size()
                    else:
                        _, C, h, w = self.lq.size()
                        split_token_h = h // 200 + 1  # number of horizontal cut sections
                        split_token_w = w // 200 + 1  # number of vertical cut sections

                        patch_size1 = max(self.opt['network_g']['split_size_0'])
                        patch_size2 = max(self.opt['network_g']['split_size_1'])
                        patch_size = max(patch_size1, patch_size2)

                        patch_size_tmp_h = split_token_h * patch_size
                        patch_size_tmp_w = split_token_w * patch_size

                        # padding
                        mod_pad_h, mod_pad_w = 0, 0
                        if h % patch_size_tmp_h != 0:
                            mod_pad_h = patch_size_tmp_h - h % patch_size_tmp_h
                        if w % patch_size_tmp_w != 0:
                            mod_pad_w = patch_size_tmp_w - w % patch_size_tmp_w

                        img = self.lq
                        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + mod_pad_h, :]
                        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]

                        _, _, H, W = img.size()
                        split_h = H // split_token_h  # height of each partition
                        split_w = W // split_token_w  # width of each partition

                        # overlapping
                        shave_h = 16
                        shave_w = 16
                        scale = self.opt.get('scale', 1)
                        ral = H // split_h
                        row = W // split_w
                        slices = []  # list of partition borders
                        for i in range(ral):
                            for j in range(row):
                                if i == 0 and i == ral - 1:
                                    top = slice(i * split_h, (i + 1) * split_h)
                                elif i == 0:
                                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                                elif i == ral - 1:
                                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                                else:
                                    top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)
                                if j == 0 and j == row - 1:
                                    left = slice(j * split_w, (j + 1) * split_w)
                                elif j == 0:
                                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                                elif j == row - 1:
                                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                                else:
                                    left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)
                                temp = (top, left)
                                slices.append(temp)
                        img_chops = []  # list of partitions
                        for temp in slices:
                            top, left = temp
                            img_chops.append(img[..., top, left])
                        if hasattr(self, 'net_g_ema'):
                            self.net_g_ema.eval()
                            with torch.no_grad():
                                outputs = []
                                for chop in img_chops:
                                    out = self.net_g_ema(chop)  # image processing of each partition
                                    outputs.append(out)
                                _img = torch.zeros(1, C, H * scale, W * scale)
                                # merge
                                for i in range(ral):
                                    for j in range(row):
                                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                                        if i == 0:
                                            _top = slice(0, split_h * scale)
                                        else:
                                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                                        if j == 0:
                                            _left = slice(0, split_w * scale)
                                        else:
                                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                                pred = _img
                        else:
                            self.net_g.eval()
                            with torch.no_grad():
                                outputs = []
                                for chop in img_chops:
                                    out = self.net_g(chop)  # image processing of each partition
                                    outputs.append(out)
                                _img = torch.zeros(1, C, H * scale, W * scale)
                                # merge
                                for i in range(ral):
                                    for j in range(row):
                                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                                        if i == 0:
                                            _top = slice(0, split_h * scale)
                                        else:
                                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                                        if j == 0:
                                            _left = slice(0, split_w * scale)
                                        else:
                                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                                pred = _img
                            self.net_g.train()
                        _, _, h, w = pred.size()
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i1 = j1
            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            print(val_data)

            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]

            self.feed_data(val_data)
            # self.feed_data_(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(self.opt['path']['visualization'], f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(self.opt['path']['visualization'], f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], str(current_iter), f'{img_name}.png')
                        save_gt_img_path = osp.join(self.opt['path']['visualization'], str(current_iter), f'{img_name}_gt.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], f'{img_name}.png')
                        save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    # imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

            keys = []
            metrics = []
            for name, value in self.collected_metrics.items():
                keys.append(name)
                metrics.append(value)
            metrics = torch.stack(metrics, 0)
            try:
                torch.distributed.reduce(metrics, dst=0)
            except:
                pass
            if self.opt['rank'] == 0:
                metrics_dict = {}
                cnt = 0
                for key, metric in zip(keys, metrics):
                    if key == 'cnt':
                        cnt = float(metric)
                        continue
                    metrics_dict[key] = float(metric)

                for key in metrics_dict:
                    metrics_dict[key] /= cnt

                self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                                   tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger, metric_dict):
        print(' ********************',dataset_name)
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in metric_dict.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
        #tb_logger.add_scalar(f'metrics/{dataset_name}/m_{metric}', value, current_iter)


        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value
            
        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


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