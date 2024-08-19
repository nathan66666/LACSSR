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

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

# class ImageRestorationModel_(BaseModel):
#     """Base Deblur model for single image deblur."""

#     def __init__(self, opt):
#         super(ImageRestorationModel_, self).__init__(opt)

#         # define network
#         self.net_g = define_network(deepcopy(opt['network_g']))
#         self.net_g = self.model_to_device(self.net_g)

#         # load pretrained models
#         load_path = self.opt['path'].get('pretrain_network_g', None)
#         if load_path is not None:
#             self.load_network(self.net_g, load_path,
#                               self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

#         if self.is_train:
#             self.init_training_settings()

#         self.scale = int(opt['scale'])

#     def init_training_settings(self):
#         self.net_g.train()
#         train_opt = self.opt['train']

#         # define losses
#         if train_opt.get('pixel_opt'):
#             pixel_type = train_opt['pixel_opt'].pop('type')
#             cri_pix_cls = getattr(loss_module, pixel_type)
#             self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
#                 self.device)
#         else:
#             self.cri_pix = None

#         if train_opt.get('perceptual_opt'):
#             percep_type = train_opt['perceptual_opt'].pop('type')
#             cri_perceptual_cls = getattr(loss_module, percep_type)
#             self.cri_perceptual = cri_perceptual_cls(
#                 **train_opt['perceptual_opt']).to(self.device)
#         else:
#             self.cri_perceptual = None

#         if self.cri_pix is None and self.cri_perceptual is None:
#             raise ValueError('Both pixel and perceptual losses are None.')

#         # set up optimizers and schedulers
#         self.setup_optimizers()
#         self.setup_schedulers()

#     def setup_optimizers(self):
#         train_opt = self.opt['train']
#         optim_params = []

#         for k, v in self.net_g.named_parameters():
#             if v.requires_grad:
#         #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
#         #             optim_params_lowlr.append(v)
#         #         else:
#                 optim_params.append(v)
#             # else:
#             #     logger = get_root_logger()
#             #     logger.warning(f'Params {k} will not be optimized.')
#         # print(optim_params)
#         # ratio = 0.1

#         optim_type = train_opt['optim_g'].pop('type')
#         if optim_type == 'Adam':
#             self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
#                                                 **train_opt['optim_g'])
#         elif optim_type == 'SGD':
#             self.optimizer_g = torch.optim.SGD(optim_params,
#                                                **train_opt['optim_g'])
#         elif optim_type == 'AdamW':
#             self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
#                                                 **train_opt['optim_g'])
#             pass
#         else:
#             raise NotImplementedError(
#                 f'optimizer {optim_type} is not supperted yet.')
#         self.optimizers.append(self.optimizer_g)

#     def feed_data(self, data, is_val=False):
#         self.lq = data['lq'].to(self.device)
#         if 'gt' in data:
#             self.gt = data['gt'].to(self.device)

#     def grids(self):
#         b, c, h, w = self.gt.size()
#         self.original_size = (b, c, h, w)

#         assert b == 1
#         if 'crop_size_h' in self.opt['val']:
#             crop_size_h = self.opt['val']['crop_size_h']
#         else:
#             crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

#         if 'crop_size_w' in self.opt['val']:
#             crop_size_w = self.opt['val'].get('crop_size_w')
#         else:
#             crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


#         crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
#         #adaptive step_i, step_j
#         num_row = (h - 1) // crop_size_h + 1
#         num_col = (w - 1) // crop_size_w + 1

#         import math
#         step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
#         step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

#         scale = self.scale
#         step_i = step_i//scale*scale
#         step_j = step_j//scale*scale

#         parts = []
#         idxes = []

#         i = 0  # 0~h-1
#         last_i = False
#         while i < h and not last_i:
#             j = 0
#             if i + crop_size_h >= h:
#                 i = h - crop_size_h
#                 last_i = True

#             last_j = False
#             while j < w and not last_j:
#                 if j + crop_size_w >= w:
#                     j = w - crop_size_w
#                     last_j = True
#                 parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
#                 idxes.append({'i': i, 'j': j})
#                 j = j + step_j
#             i = i + step_i

#         self.origin_lq = self.lq
#         self.lq = torch.cat(parts, dim=0)
#         self.idxes = idxes

#     def grids_inverse(self):
#         preds = torch.zeros(self.original_size)
#         b, c, h, w = self.original_size

#         count_mt = torch.zeros((b, 1, h, w))
#         if 'crop_size_h' in self.opt['val']:
#             crop_size_h = self.opt['val']['crop_size_h']
#         else:
#             crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

#         if 'crop_size_w' in self.opt['val']:
#             crop_size_w = self.opt['val'].get('crop_size_w')
#         else:
#             crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

#         crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

#         for cnt, each_idx in enumerate(self.idxes):
#             i = each_idx['i']
#             j = each_idx['j']
#             preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
#             count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

#         self.output = (preds / count_mt).to(self.device)
#         self.lq = self.origin_lq

#     def optimize_parameters(self, current_iter, tb_logger):
#         self.optimizer_g.zero_grad()

#         if self.opt['train'].get('mixup', False):
#             self.mixup_aug()

#         preds = self.net_g(self.lq)
#         if not isinstance(preds, list):
#             preds = [preds]

#         self.output = preds[-1]

#         l_total = 0
#         loss_dict = OrderedDict()
#         # pixel loss
#         if self.cri_pix:
#             l_pix = 0.
#             for pred in preds:
#                 l_pix += self.cri_pix(pred, self.gt)

#             # print('l pix ... ', l_pix)
#             l_total += l_pix
#             loss_dict['l_pix'] = l_pix

#         # perceptual loss
#         if self.cri_perceptual:
#             l_percep, l_style = self.cri_perceptual(self.output, self.gt)
#         #
#             if l_percep is not None:
#                 l_total += l_percep
#                 loss_dict['l_percep'] = l_percep
#             if l_style is not None:
#                 l_total += l_style
#                 loss_dict['l_style'] = l_style


#         l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

#         l_total.backward()
#         use_grad_clip = self.opt['train'].get('use_grad_clip', True)
#         if use_grad_clip:
#             torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
#         self.optimizer_g.step()


#         self.log_dict = self.reduce_loss_dict(loss_dict)

#     def test(self):
#         self.net_g.eval()
#         with torch.no_grad():
#             n = len(self.lq)
#             outs = []
#             m = self.opt['val'].get('max_minibatch', n)
#             i = 0
#             while i < n:
#                 j = i + m
#                 if j >= n:
#                     j = n
#                 pred = self.net_g(self.lq[i:j])
#                 if isinstance(pred, list):
#                     pred = pred[-1]
#                 outs.append(pred.detach().cpu())
#                 i = j

#             self.output = torch.cat(outs, dim=0)
#         self.net_g.train()

#     def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
#         dataset_name = dataloader.dataset.opt['name']
#         with_metrics = self.opt['val'].get('metrics') is not None
#         if with_metrics:
#             self.metric_results = {
#                 metric: 0
#                 for metric in self.opt['val']['metrics'].keys()
#             }

#         rank, world_size = get_dist_info()
#         if rank == 0:
#             pbar = tqdm(total=len(dataloader), unit='image')

#         cnt = 0

#         for idx, val_data in enumerate(dataloader):
#             if idx % world_size != rank:
#                 continue

#             img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

#             self.feed_data(val_data, is_val=True)
#             if self.opt['val'].get('grids', False):
#                 self.grids()

#             self.test()

#             if self.opt['val'].get('grids', False):
#                 self.grids_inverse()

#             visuals = self.get_current_visuals()
#             sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
#             if 'gt' in visuals:
#                 gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
#                 del self.gt

#             # tentative for out of GPU memory
#             del self.lq
#             del self.output
#             torch.cuda.empty_cache()

#             if save_img:
#                 if sr_img.shape[2] == 6:
#                     L_img = sr_img[:, :, :3]
#                     R_img = sr_img[:, :, 3:]

#                     # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
#                     visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

#                     imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
#                     imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
#                 else:
#                     if self.opt['is_train']:

#                         save_img_path = osp.join(self.opt['path']['visualization'],
#                                                  img_name,
#                                                  f'{img_name}_{current_iter}.png')

#                         save_gt_img_path = osp.join(self.opt['path']['visualization'],
#                                                  img_name,
#                                                  f'{img_name}_{current_iter}_gt.png')
#                     else:
#                         save_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}.png')
#                         save_gt_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}_gt.png')

#                     imwrite(sr_img, save_img_path)
#                     imwrite(gt_img, save_gt_img_path)

#             if with_metrics:
#                 # calculate metrics
#                 opt_metric = deepcopy(self.opt['val']['metrics'])
#                 if use_image:
#                     for name, opt_ in opt_metric.items():
#                         metric_type = opt_.pop('type')
#                         self.metric_results[name] += getattr(
#                             metric_module, metric_type)(sr_img, gt_img, **opt_)
#                 else:
#                     for name, opt_ in opt_metric.items():
#                         metric_type = opt_.pop('type')
#                         self.metric_results[name] += getattr(
#                             metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

#             cnt += 1
#             if rank == 0:
#                 for _ in range(world_size):
#                     pbar.update(1)
#                     pbar.set_description(f'Test {img_name}')
#         if rank == 0:
#             pbar.close()

#         # current_metric = 0.
#         collected_metrics = OrderedDict()
#         if with_metrics:
#             for metric in self.metric_results.keys():
#                 collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
#             collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

#             self.collected_metrics = collected_metrics
        
#         keys = []
#         metrics = []
#         for name, value in self.collected_metrics.items():
#             keys.append(name)
#             metrics.append(value)
#         metrics = torch.stack(metrics, 0)
#         torch.distributed.reduce(metrics, dst=0)
#         if self.opt['rank'] == 0:
#             metrics_dict = {}
#             cnt = 0
#             for key, metric in zip(keys, metrics):
#                 if key == 'cnt':
#                     cnt = float(metric)
#                     continue
#                 metrics_dict[key] = float(metric)

#             for key in metrics_dict:
#                 metrics_dict[key] /= cnt

#             self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
#                                                tb_logger, metrics_dict)
#         return 0.

#     def nondist_validation(self, *args, **kwargs):
#         logger = get_root_logger()
#         logger.warning('nondist_validation is not implemented. Run dist_validation.')
#         self.dist_validation(*args, **kwargs)


#     def _log_validation_metric_values(self, current_iter, dataset_name,
#                                       tb_logger, metric_dict):
#         log_str = f'Validation {dataset_name}, \t'
#         for metric, value in metric_dict.items():
#             log_str += f'\t # {metric}: {value:.4f}'
#         logger = get_root_logger()
#         logger.info(log_str)

#         log_dict = OrderedDict()
#         # for name, value in loss_dict.items():
#         for metric, value in metric_dict.items():
#             log_dict[f'm_{metric}'] = value

#         self.log_dict = log_dict

#     def get_current_visuals(self):
#         out_dict = OrderedDict()
#         out_dict['lq'] = self.lq.detach().cpu()
#         out_dict['result'] = self.output.detach().cpu()
#         if hasattr(self, 'gt'):
#             out_dict['gt'] = self.gt.detach().cpu()
#         return out_dict

#     def save(self, epoch, current_iter):
#         self.save_network(self.net_g, 'net_g', current_iter)
#         self.save_training_state(epoch, current_iter)

class ImageRestorationModel_(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel_, self).__init__(opt)

        # define network
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

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

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
                    if self.opt['network_g']['type'] == 'NAFSSR' :
                        img = self.lq  # img
                        pred = self.net_g(img)
                    else:
                        if self.opt['network_g']['type'] == 'NAFSSR' : 
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
                        if self.opt['network_g']['type'] == 'STSSR' or self.opt['network_g']['type'] == 'STSSR_SSCAM_o' or self.opt['network_g']['type'] == 'SwinIR' or self.opt['network_g']['type'] == 'STSSR_MSCAM_o' or self.opt['network_g']['type'] == 'STSSR_SCAM':
                            window_size = self.opt['network_g']['window_size']
                            scale = self.opt.get('scale', 1)
                            mod_pad_h, mod_pad_w = 0, 0
                            _, _, h, w = self.lq.size()
                            if h % window_size != 0:
                                mod_pad_h = window_size - h % window_size
                            if w % window_size != 0:
                                mod_pad_w = window_size - w % window_size

                        if self.opt['network_g']['type'] == 'NAFSSR_DSCAM_o' or self.opt['network_g']['type'] == 'NAFSSR_DSCAM' or self.opt['network_g']['type'] == 'STSSRHA' or self.opt['network_g']['type'] == 'DSCTSR' or self.opt['network_g']['type'] == 'PSASR' or self.opt['network_g']['type'] == 'SWIN2SR' or self.opt['network_g']['type'] == 'ST2SSR_OCAB_MSCAM_1' or self.opt['network_g']['type'] == 'ST2SSR_OCAB_MSCAM_2' or self.opt['network_g']['type'] == 'ST2SSR_OCAB':
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

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            # self.feed_data(val_data, is_val=True)
            self.feed_data(val_data)
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

                    imwrite(L_img, osp.join(self.opt['path']['visualization'], f'{img_name}_L_{self.opt["name"]}.png'))
                    imwrite(R_img, osp.join(self.opt['path']['visualization'], f'{img_name}_R_{self.opt["name"]}.png'))
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
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in metric_dict.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

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


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

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

    def feed_data_(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

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
        load_path_L = "/lxy/NAFNet/experiments/xs000_GAN_DSCAM-ST_0.5/models/net_g_185000.pth"
        self.net_g.load_state_dict(torch.load(load_path_L)["params"], strict=False)
        # load_path_Lora = "/lxy/NAFNet/experiments/LACSSR_lora/models/net_g_5000.pth"
        # self.net_g.load_state_dict(torch.load(load_path_Lora), strict=False)
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
                    if self.opt['network_g']['type'] == 'NAFSSR' or self.opt['network_g']['type'] == 'RRDBNet' or self.opt['network_g']['type'] == 'SSRDEFNet' or self.opt['network_g']['type'] == 'CVHSSR':
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
                        if self.opt['network_g']['type'] == 'STSSR' or self.opt['network_g']['type'] == 'NAFSSRP' or self.opt['network_g']['type'] == 'SwinIR' or self.opt['network_g']['type'] == 'HATSSR':
                            window_size = self.opt['network_g']['window_size']
                            scale = self.opt.get('scale', 1)
                            mod_pad_h, mod_pad_w = 0, 0
                            _, _, h, w = self.lq.size()
                            if h % window_size != 0:
                                mod_pad_h = window_size - h % window_size
                            if w % window_size != 0:
                                mod_pad_w = window_size - w % window_size

                        if self.opt['network_g']['type'] == 'Swin2SR' or self.opt['network_g']['type'] == 'NAFSSR_MSCAM' or self.opt['network_g']['type'] == 'ST2SSR_MSCAM' or self.opt['network_g']['type'] == 'SWIN2SR' or self.opt['network_g']['type'] == 'ST2SSR_MSCAM_' or self.opt['network_g']['type'] == 'MSTSSR' or self.opt['network_g']['type'] =='STSSR_MSCAM' or self.opt['network_g']['type'] =='LACSSR_lora':
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
                        # load_path_L = "/lxy/NAFNet/experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
                        # self.net_g.load_state_dict(torch.load(load_path_L), strict=False)
                        # load_path_Lora = "/lxy/NAFNet/experiments/LACSSR/models/net_g_9000.pth"
                        # self.net_g.load_state_dict(torch.load(load_path_Lora), strict=False)
                        # print(self.net_g)
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

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            # self.feed_data(val_data, is_val=True)
            self.feed_data_(val_data)
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
                    # imwrite(L_img, osp.join(self.opt['path']['visualization'], f'{img_name}_L_{self.opt["name"]}.png'))
                    # imwrite(R_img, osp.join(self.opt['path']['visualization'], f'{img_name}_R_{self.opt["name"]}.png'))

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L_{self.opt["name"]}.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R_{self.opt["name"]}.png'))
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
        #print('#####################',dataset_name)
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in metric_dict.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
       

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
