import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from basicsr.models.losses.loss_util import get_refined_artifact_map
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.models.srgan_model import SRGANModel


from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class ImageRestoration_GAN_Model(SRGANModel):
    def __init__(self, opt):
        super(ImageRestoration_GAN_Model, self).__init__(opt)

        # # define network
        # self.net_g = define_network(deepcopy(opt['network_g']))
        # self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # # load pretrained models
        # load_path = self.opt['path'].get('pretrain_network_g', None)
        # if load_path is not None:
        #     self.load_network(self.net_g, load_path,
        #                       self.opt['path'].get('strict_load_g', False), param_key=self.opt['path'].get('param_key', 'params'))

        # if self.is_train:
        #     self.init_training_settings()

        # self.scale = int(opt['scale'])

   
    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(ImageRestoration_GAN_Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # # usm sharpening
        # l1_gt = self.gt_usm
        # percep_gt = self.gt_usm
        # gan_gt = self.gt_usm
        # if self.opt['l1_gt_usm'] is False:
        #     l1_gt = self.gt
        # if self.opt['percep_gt_usm'] is False:
        #     percep_gt = self.gt
        # if self.opt['gan_gt_usm'] is False:
        #     gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]

        if self.cri_ldl:
            self.output_emas = self.net_g_ema(self.lq)
            if not isinstance(self.output_emas, list):
                self.output_emas = [self.output_emas]


        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = 0.
                for pred in preds:
                    l_g_pix += self.cri_pix(pred, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            if self.cri_ldl:
                l_g_total = 0.
                for pred in preds:
                    pixel_weight = get_refined_artifact_map(self.gt, pred, self.output_ema[i], 7)
                    l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, pred), torch.mul(pixel_weight, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep = 0.
                l_g_style = 0.
                for pred in preds:
                    pred = pred.chunk(2,dim=1)
                    self.gt = (self.gt).chunk(2,dim=1)
                    i=0
                    #print(pred.shape)
                    for x in list(pred):
                        if self.cri_perceptual(x, (self.gt)[i])[0] is not None:
                            l_g_percep += self.cri_perceptual(x, (self.gt)[i])[0]
                        if self.cri_perceptual(x, (self.gt)[i])[1] is not None:
                            l_g_style += self.cri_perceptual(x, (self.gt)[i])[1]
                        i+=1
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    if l_g_percep != 0:
                        loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    if l_g_style != 0.:
                        loss_dict['l_g_style'] = l_g_style
            # gan loss
            l_g_gan = 0.
            for pred in preds:
                pred = pred.chunk(2,dim=1)
                for x in pred:
                    fake_g_pred = self.net_d(x)
                    l_g_gan += self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            if l_g_gan != 0.:
                loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        #print((self.gt)[0].shape)
        l_d_real=0.
        for x in self.gt:
            real_d_pred = self.net_d(x)
            l_d_real += self.cri_gan(real_d_pred, True, is_disc=True)
        if l_d_real!=0.:
            loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        
        for pred in preds:
            l_d_fake=0.
            pred = pred.chunk(2,dim=1)
            for x in pred:
                fake_d_pred = self.net_d(x.detach().clone())  # clone for pt1.9
                l_d_fake += self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)