import torch
from collections import OrderedDict
import importlib
from copy import deepcopy
import os
from basicsr.models.archs import define_network
# from basicsr.archs import build_network
# from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
#from basicsr.utils.registry import MODEL_REGISTRY
from .sr_lora_model import SR_lora_Model
import loralib as lora
import time
from typing import Dict
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
# @MODEL_REGISTRY.register()
class SRGAN_lora_Model(SR_lora_Model):
    """SRGAN model for single image super-resolution."""

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(deepcopy(self.opt['network_g'])).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # self.net_g = define_network(deepcopy(self.opt['LACSSR_net']))
        # self.net_g = self.model_to_device(self.net_g)

        # define network net_d
        self.net_d = define_network(deepcopy(self.opt['network_d']))
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # # define network net_g
        # self.net_g = define_network(deepcopy(self.opt['network_g']))
        # self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # # load pretrained models
        # load_path_g = self.opt['path'].get('pretrain_network_g', None)
        # if load_path is not None:
        #     param_key = self.opt['path'].get('param_key_g', 'params')
        #     self.load_network(self.net_g, load_path_g, self.opt['path'].get('strict_load_g', True), param_key)

        # # load pretrained models
        # L_load_path = self.opt['path'].get('LACSSR_net_path', None)
        # if L_load_path is not None:
        #     param_key = self.opt['path'].get('param_key_L', 'params')
        #     self.load_network(self.net_g, L_load_path, self.opt['path'].get('strict_load_L', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('ldl_opt'):
            ldl_type = train_opt['ldl_opt'].pop('type')
            cri_ldl_cls = getattr(loss_module, ldl_type)
            self.cri_ldl = cri_ldl_cls(**train_opt['ldl_opt']).to(
                self.device)
        else:
            self.cri_ldl = None

        if train_opt.get('perceptual_opt'):
            perceptual_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, perceptual_type)
            self.cri_perceptual = cri_perceptual_cls(**train_opt['perceptual_opt']).to(
                self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(
                self.device)
        else:
            self.cri_gan = None


        # if train_opt.get('gan_opt'):
        #     self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

         # lora setting
        lora.mark_only_lora_as_trainable(self.net_g)
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network_lora(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
    def save_network_lora(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                # torch.save(save_dict, save_path)
                torch.save(self.lora_state_dict(save_dict['params']), save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

    def lora_state_dict(self, my_state_dict , bias: str = 'none') -> Dict[str, torch.Tensor]:

        if bias == 'none':
            return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
        elif bias == 'all':
            return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
        elif bias == 'lora_only':
            to_return = {}
            for k in my_state_dict:
                if 'lora_' in k:
                    to_return[k] = my_state_dict[k]
                    bias_name = k.split('lora_')[0]+'bias'
                    if bias_name in my_state_dict:
                        to_return[bias_name] = my_state_dict[bias_name]
            return to_return
        else:
            raise NotImplementedError




