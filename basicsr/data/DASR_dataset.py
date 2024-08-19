from torch.utils import data as data
from torchvision.transforms.functional import normalize, resize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, gt_random_crop_hw
from basicsr.data.degradations_dasr import circular_lowpass_kernel, random_mixed_kernels_Info
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, scandir,get_root_logger
import os
import time
import numpy as np
#from basicsr.data import degrations as degrations
from basicsr.data.degradations import random_mixed_kernels
import random
import math
import torch
import cv2

class DASR_Dataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''

    def __init__(self, opt):
        super(DASR_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        
        
        import os
        self.gt_files = os.listdir(self.gt_folder)
        self.nums = len(self.gt_files)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.blur_kernel_size_minimum = opt['blur_kernel_size_minimum']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.blur_kernel_size2_minimum = opt['blur_kernel_size2_minimum']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(math.ceil(self.blur_kernel_size_minimum / 2), math.ceil(self.blur_kernel_size / 2))]
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.kernel_range2 = [2 * v + 1 for v in range(math.ceil(self.blur_kernel_size2_minimum / 2), math.ceil(self.blur_kernel_size2 / 2))]
        self.pulse_tensor2 = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor2[10, 10] = 1

        # standard_degrade_one_stage
        self.blur_kernel_size_standard1 = opt['blur_kernel_size_standard1']
        self.blur_kernel_size_minimum_standard1 = opt['blur_kernel_size_minimum_standard1']
        self.kernel_list_standard1 = opt['kernel_list_standard1']
        self.kernel_prob_standard1 = opt['kernel_prob_standard1']
        self.blur_sigma_standard1 = opt['blur_sigma_standard1']
        self.betag_range_standard1 = opt['betag_range_standard1']
        self.betap_range_standard1 = opt['betap_range_standard1']
        self.sinc_prob_standard1 = opt['sinc_prob_standard1']

        # weak_degrade_one_stage
        self.blur_kernel_size_weak1 = opt['blur_kernel_size_weak1']
        self.blur_kernel_size_minimum_weak1 = opt['blur_kernel_size_minimum_weak1']
        self.kernel_list_weak1 = opt['kernel_list_weak1']
        self.kernel_prob_weak1 = opt['kernel_prob_weak1']
        self.blur_sigma_weak1 = opt['blur_sigma_weak1']
        self.betag_range_weak1 = opt['betag_range_weak1']
        self.betap_range_weak1 = opt['betap_range_weak1']
        self.sinc_prob_weak1 = opt['sinc_prob_weak1']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')


        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))
        
        #print('0',img_gt_R.shape)

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)

        gt_path = os.path.join(self.gt_folder, self.gt_files[index])

        
        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size


            # if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
            #     idx = [
            #         [0, 1, 2, 3, 4, 5],
            #         [0, 2, 1, 3, 5, 4],
            #         [1, 0, 2, 4, 3, 5],
            #         [1, 2, 0, 4, 5, 3],
            #         [2, 0, 1, 5, 3, 4],
            #         [2, 1, 0, 5, 4, 3],
            #     ][int(np.random.rand() * 6)]

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2],
                    [1, 0, 2],
                    [1, 2, 0],
                ][int(np.random.rand() * 3)]

                img_gt_L = img_gt_L[:, :, idx]
                img_gt_R = img_gt_R[:, :, idx]
     

            # random crop
            img_gt_L = img_gt_L.copy()
            img_gt_L = gt_random_crop_hw(img_gt_L, gt_size_h, gt_size_w, scale, 'gt_path_L')
            # flip, rotation
            imgs_L, status = augment(img_gt_L, self.opt['use_hflip'], self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)

            img_gt_L = imgs_L

             # random crop
            img_gt_R = img_gt_R.copy()
            img_gt_R = gt_random_crop_hw(img_gt_R, gt_size_h, gt_size_w, scale, 'gt_path_R')
            # flip, rotation
            imgs_R, status = augment(img_gt_R, self.opt['use_hflip'], self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)

            img_gt_R = imgs_R

        img_gt_L = img2tensor([img_gt_L], bgr2rgb=True, float32=True)[0]
        img_gt_R = img2tensor([img_gt_R], bgr2rgb=True, float32=True)[0]

        #print(img_gt_R.shape)
        return_d = {}

        # severe_degrade_two_stage
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2_info = random_mixed_kernels_Info(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)
            kernel2 = kernel2_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
            kernel_sinc_info = {'kernel':sinc_kernel, 'kernel_size':kernel_size, 'omega_c':omega_c}
        else:
            sinc_kernel = self.pulse_tensor2
            kernel_sinc_info = {'kernel': sinc_kernel, 'kernel_size': 0, 'omega_c': 0}

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        kernel_info['kernel'] = kernel
        kernel2_info['kernel'] = kernel2

        return_d['severe_degrade_two_stage'] = {'gt_L': img_gt_L, 'gt_R': img_gt_R, 'kernel1': kernel_info, 'kernel2': kernel2_info, 'sinc_kernel': kernel_sinc_info, 'gt_path': gt_path}
        # return_d = {'gt': img_gt, 'kernel1': kernel_info, 'kernel2': kernel2_info, 'sinc_kernel': kernel_sinc_info, 'gt_path': gt_path}

        kernel_info = {}

        # standard_degrade_one_stage

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob_standard1']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list_standard1,
                self.kernel_prob_standard1,
                kernel_size,
                self.blur_sigma_standard1,
                self.blur_sigma_standard1, [-math.pi, math.pi],
                self.betag_range_standard1,
                self.betap_range_standard1,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel = torch.FloatTensor(kernel)
        kernel_info['kernel'] = kernel

        return_d['standard_degrade_one_stage'] = {'gt_L': img_gt_L, 'gt_R': img_gt_R,'kernel1': kernel_info, 'gt_path': gt_path}

        kernel_info = {}

        # weak_degrade_one_stage

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob_weak1']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list_weak1,
                self.kernel_prob_weak1,
                kernel_size,
                self.blur_sigma_weak1,
                self.blur_sigma_weak1, [-math.pi, math.pi],
                self.betag_range_weak1,
                self.betap_range_weak1,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel = torch.FloatTensor(kernel)
        kernel_info['kernel'] = kernel
#torch.cat([img_gt_L, img_gt_R], dim=0)
        return_d['weak_degrade_one_stage'] = {'gt_L': img_gt_L, 'gt_R':img_gt_R, 'kernel1': kernel_info, 'gt_path': gt_path}

        return return_d


    def __len__(self):
        return self.nums

class DASR_bic_Dataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''

    def __init__(self, opt):
        super(DASR_bic_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        
        
        import os
        self.gt_files = os.listdir(self.gt_folder)
        self.nums = len(self.gt_files)

         # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.blur_kernel_size_minimum = opt['blur_kernel_size_minimum']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.blur_kernel_size2_minimum = opt['blur_kernel_size2_minimum']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(math.ceil(self.blur_kernel_size_minimum / 2), math.ceil(self.blur_kernel_size / 2))]
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.kernel_range2 = [2 * v + 1 for v in range(math.ceil(self.blur_kernel_size2_minimum / 2), math.ceil(self.blur_kernel_size2 / 2))]
        self.pulse_tensor2 = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor2[10, 10] = 1

        # standard_degrade_one_stage
        self.blur_kernel_size_standard1 = opt['blur_kernel_size_standard1']
        self.blur_kernel_size_minimum_standard1 = opt['blur_kernel_size_minimum_standard1']
        self.kernel_list_standard1 = opt['kernel_list_standard1']
        self.kernel_prob_standard1 = opt['kernel_prob_standard1']
        self.blur_sigma_standard1 = opt['blur_sigma_standard1']
        self.betag_range_standard1 = opt['betag_range_standard1']
        self.betap_range_standard1 = opt['betap_range_standard1']
        self.sinc_prob_standard1 = opt['sinc_prob_standard1']

        # weak_degrade_one_stage
        self.blur_kernel_size_weak1 = opt['blur_kernel_size_weak1']
        self.blur_kernel_size_minimum_weak1 = opt['blur_kernel_size_minimum_weak1']
        self.kernel_list_weak1 = opt['kernel_list_weak1']
        self.kernel_prob_weak1 = opt['kernel_prob_weak1']
        self.blur_sigma_weak1 = opt['blur_sigma_weak1']
        self.betag_range_weak1 = opt['betag_range_weak1']
        self.betap_range_weak1 = opt['betap_range_weak1']
        self.sinc_prob_weak1 = opt['sinc_prob_weak1']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
        gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')


        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))
        
        #print('0',img_gt_R.shape)

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)

        gt_path = os.path.join(self.gt_folder, self.gt_files[index])

        
        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2],
                    [1, 0, 2],
                    [1, 2, 0],
                ][int(np.random.rand() * 3)]

                img_gt_L = img_gt_L[:, :, idx]
                img_gt_R = img_gt_R[:, :, idx]
     

            # random crop
            img_gt_L = img_gt_L.copy()
            img_gt_L = gt_random_crop_hw(img_gt_L, gt_size_h, gt_size_w, scale, 'gt_path_L')
            # flip, rotation
            imgs_L, status = augment(img_gt_L, self.opt['use_hflip'], self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)

            img_gt_L = imgs_L

             # random crop
            img_gt_R = img_gt_R.copy()
            img_gt_R = gt_random_crop_hw(img_gt_R, gt_size_h, gt_size_w, scale, 'gt_path_R')
            # flip, rotation
            imgs_R, status = augment(img_gt_R, self.opt['use_hflip'], self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)

            img_gt_R = imgs_R

        img_gt_L = img2tensor([img_gt_L], bgr2rgb=True, float32=True)[0]
        img_gt_R = img2tensor([img_gt_R], bgr2rgb=True, float32=True)[0]

        #print(img_gt_R.shape)
        return_d = {}
        # severe_degrade_two_stage
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2_info = random_mixed_kernels_Info(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)
            kernel2 = kernel2_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
            kernel_sinc_info = {'kernel':sinc_kernel, 'kernel_size':kernel_size, 'omega_c':omega_c}
        else:
            sinc_kernel = self.pulse_tensor2
            kernel_sinc_info = {'kernel': sinc_kernel, 'kernel_size': 0, 'omega_c': 0}

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        kernel_info['kernel'] = kernel
        kernel2_info['kernel'] = kernel2

        return_d['severe_degrade_two_stage'] = {'gt_L': img_gt_L, 'gt_R': img_gt_R, 'kernel1': kernel_info, 'kernel2': kernel2_info, 'sinc_kernel': kernel_sinc_info, 'gt_path': gt_path}
        # return_d = {'gt': img_gt, 'kernel1': kernel_info, 'kernel2': kernel2_info, 'sinc_kernel': kernel_sinc_info, 'gt_path': gt_path}

        kernel_info = {}


        # standard_degrade_one_stage

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob_standard1']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list_standard1,
                self.kernel_prob_standard1,
                kernel_size,
                self.blur_sigma_standard1,
                self.blur_sigma_standard1, [-math.pi, math.pi],
                self.betag_range_standard1,
                self.betap_range_standard1,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel = torch.FloatTensor(kernel)
        kernel_info['kernel'] = kernel

        return_d['standard_degrade_one_stage'] = {'gt_L': img_gt_L, 'gt_R': img_gt_R,'kernel1': kernel_info, 'gt_path': gt_path}

        kernel_info = {}

        # weak_degrade_one_stage

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob_weak1']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list_weak1,
                self.kernel_prob_weak1,
                kernel_size,
                self.blur_sigma_weak1,
                self.blur_sigma_weak1, [-math.pi, math.pi],
                self.betag_range_weak1,
                self.betap_range_weak1,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel = torch.FloatTensor(kernel)
        kernel_info['kernel'] = kernel
        return_d['weak_degrade_one_stage'] = {'gt_L': img_gt_L, 'gt_R':img_gt_R, 'kernel1': kernel_info, 'gt_path': gt_path}

        
        return_d['bic'] = {'gt_L': img_gt_L, 'gt_R':img_gt_R, 'gt_path': gt_path}
        
        return return_d


    def __len__(self):
        return self.nums


class DASR_Dataset1(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''

    def __init__(self, opt):
        super(DASR_Dataset1, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        self.gt_folder1 = opt['dataroot_gt1']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        
        self.img=[]
        import os
        self.gt_files = (os.listdir(self.gt_folder))
        #print(type(self.gt_files))
        self.gt_files1 = (os.listdir(self.gt_folder1))

        for i in range(len(self.gt_files)):
            self.img.append(os.path.join(self.gt_folder,self.gt_files[i]))

        for i in range(len(self.gt_files1)):
            self.img.append(os.path.join(self.gt_folder1,self.gt_files1[i]))



        self.nums = len(self.img)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.blur_kernel_size_minimum = opt['blur_kernel_size_minimum']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.blur_kernel_size2_minimum = opt['blur_kernel_size2_minimum']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(math.ceil(self.blur_kernel_size_minimum / 2), math.ceil(self.blur_kernel_size / 2))]
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.kernel_range2 = [2 * v + 1 for v in range(math.ceil(self.blur_kernel_size2_minimum / 2), math.ceil(self.blur_kernel_size2 / 2))]
        self.pulse_tensor2 = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor2[10, 10] = 1

        # standard_degrade_one_stage
        self.blur_kernel_size_standard1 = opt['blur_kernel_size_standard1']
        self.blur_kernel_size_minimum_standard1 = opt['blur_kernel_size_minimum_standard1']
        self.kernel_list_standard1 = opt['kernel_list_standard1']
        self.kernel_prob_standard1 = opt['kernel_prob_standard1']
        self.blur_sigma_standard1 = opt['blur_sigma_standard1']
        self.betag_range_standard1 = opt['betag_range_standard1']
        self.betap_range_standard1 = opt['betap_range_standard1']
        self.sinc_prob_standard1 = opt['sinc_prob_standard1']

        # weak_degrade_one_stage
        self.blur_kernel_size_weak1 = opt['blur_kernel_size_weak1']
        self.blur_kernel_size_minimum_weak1 = opt['blur_kernel_size_minimum_weak1']
        self.kernel_list_weak1 = opt['kernel_list_weak1']
        self.kernel_prob_weak1 = opt['kernel_prob_weak1']
        self.blur_sigma_weak1 = opt['blur_sigma_weak1']
        self.betag_range_weak1 = opt['betag_range_weak1']
        self.betap_range_weak1 = opt['betap_range_weak1']
        self.sinc_prob_weak1 = opt['sinc_prob_weak1']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
        # gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        # gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')

        gt_path_L = os.path.join(self.img[index], 'hr0.png')
        
        gt_path_R = os.path.join(self.img[index], 'hr1.png')
        #print(gt_path_L,gt_path_R)


        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))
        
        #print('0',img_gt_R.shape)

       #print(img_gt_L.shape==img_gt_R.shape)

        #img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)

        gt_path = self.img[index]

        
        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size


            # if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
            #     idx = [
            #         [0, 1, 2, 3, 4, 5],
            #         [0, 2, 1, 3, 5, 4],
            #         [1, 0, 2, 4, 3, 5],
            #         [1, 2, 0, 4, 5, 3],
            #         [2, 0, 1, 5, 3, 4],
            #         [2, 1, 0, 5, 4, 3],
            #     ][int(np.random.rand() * 6)]

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2],
                    [1, 0, 2],
                    [1, 2, 0],
                ][int(np.random.rand() * 3)]

                img_gt_L = img_gt_L[:, :, idx]
                img_gt_R = img_gt_R[:, :, idx]
     

            # random crop
            img_gt_L = img_gt_L.copy()
            img_gt_L = gt_random_crop_hw(img_gt_L, gt_size_h, gt_size_w, scale, 'gt_path_L')
            # flip, rotation
            imgs_L, status = augment(img_gt_L, self.opt['use_hflip'], self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)

            img_gt_L = imgs_L

             # random crop
            img_gt_R = img_gt_R.copy()
            img_gt_R = gt_random_crop_hw(img_gt_R, gt_size_h, gt_size_w, scale, 'gt_path_R')
            # flip, rotation
            imgs_R, status = augment(img_gt_R, self.opt['use_hflip'], self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)

            img_gt_R = imgs_R

        img_gt_L = img2tensor([img_gt_L], bgr2rgb=True, float32=True)[0]
        img_gt_R = img2tensor([img_gt_R], bgr2rgb=True, float32=True)[0]

        #print(img_gt_R.shape)
        return_d = {}

        # severe_degrade_two_stage
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2_info = random_mixed_kernels_Info(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)
            kernel2 = kernel2_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
            kernel_sinc_info = {'kernel':sinc_kernel, 'kernel_size':kernel_size, 'omega_c':omega_c}
        else:
            sinc_kernel = self.pulse_tensor2
            kernel_sinc_info = {'kernel': sinc_kernel, 'kernel_size': 0, 'omega_c': 0}

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        kernel_info['kernel'] = kernel
        kernel2_info['kernel'] = kernel2

        return_d['severe_degrade_two_stage'] = {'gt_L': img_gt_L, 'gt_R': img_gt_R, 'kernel1': kernel_info, 'kernel2': kernel2_info, 'sinc_kernel': kernel_sinc_info, 'gt_path': gt_path}
        # return_d = {'gt': img_gt, 'kernel1': kernel_info, 'kernel2': kernel2_info, 'sinc_kernel': kernel_sinc_info, 'gt_path': gt_path}

        kernel_info = {}

        # standard_degrade_one_stage

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob_standard1']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list_standard1,
                self.kernel_prob_standard1,
                kernel_size,
                self.blur_sigma_standard1,
                self.blur_sigma_standard1, [-math.pi, math.pi],
                self.betag_range_standard1,
                self.betap_range_standard1,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel = torch.FloatTensor(kernel)
        kernel_info['kernel'] = kernel
        # print(type(img_gt_L),(img_gt_L).shape)
        # from PIL import Image

        # from torchvision import transforms
        # toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值

        # pic = toPIL(img_gt_L)
        # pic.save('/lxy/random.jpg')
        

        return_d['standard_degrade_one_stage'] = {'gt_L': img_gt_L, 'gt_R': img_gt_R,'kernel1': kernel_info, 'gt_path': gt_path}

        kernel_info = {}

        # weak_degrade_one_stage

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob_weak1']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list_weak1,
                self.kernel_prob_weak1,
                kernel_size,
                self.blur_sigma_weak1,
                self.blur_sigma_weak1, [-math.pi, math.pi],
                self.betag_range_weak1,
                self.betap_range_weak1,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel = torch.FloatTensor(kernel)
        kernel_info['kernel'] = kernel
#torch.cat([img_gt_L, img_gt_R], dim=0)
        return_d['weak_degrade_one_stage'] = {'gt_L': img_gt_L, 'gt_R':img_gt_R, 'kernel1': kernel_info, 'gt_path': gt_path}

        return return_d


    def __len__(self):
        return self.nums
    

