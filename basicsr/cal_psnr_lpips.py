import cv2
import math
import numpy as np
import lpips
import torch
from PIL import Image
from basicsr.metrics.metric_util import reorder_image
from basicsr.data.transforms import totensor
import cv2
import numpy as np
import os
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from skimage.metrics import structural_similarity
import torch

def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
        
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    def _psnr(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        max_value = 1. if img1.max() <= 1 else 255.
        return 20. * np.log10(max_value / np.sqrt(mse))
    
    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:,:,:3], img1[:,:,3:]
        l2, r2 = img2[:,:,:3], img2[:,:,3:]
        return (_psnr(l1, l2) + _psnr(r1, r2))/2
    else:
        return _psnr(img1, img2)

def calculate_lpips(img1,
                    img2,
                    crop_border,
                    input_order='HWC'):
    """Calculate LPIPS metric.

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: LPIPS result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)


    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img1, img2 = totensor([img1, img2], bgr2rgb=False, float32=True)

    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = (img1 / 255. - 0.5) * 2
    img2 = (img2 / 255. - 0.5) * 2

    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False) # best forward scores

    metric = loss_fn_alex(img1, img2).squeeze(0).float().detach().cpu().numpy()
    return metric.mean()

if __name__ == "__main__":
    img1_path = '/lxy/SSRDEFNet-PyTorch/results/SSRDEF_4xSR/Flickr1024_val/'
    img2_path = '/lxy/SSRDEFNet-PyTorch/results/SSRDEF_4xSR/holopix_level1_100/'
    img3_path = '/lxy/SSRDEFNet-PyTorch/results/SSRDEF_4xSR/holopix_level2_100/'
    img4_path = '/lxy/SSRDEFNet-PyTorch/results/SSRDEF_4xSR/holopix_level3_100/'
    img5_path = '/lxy/iPASSR/results/iPASSR_holopix_bic/'
    img_path_flickr1024 = '/lxy/datasets/stereo/Val/Validation/'
    img_path_hix = '/lxy/Holopix50k/holopix_100/hr/'
    file_list = os.listdir(img5_path)
    file_list.sort()
    file_list1 = os.listdir(img_path_hix)
    file_list1.sort()
    LLPIPS=[]
    RLPIPS=[]
    LPSNR=[]
    RPSNR=[]
    print((len(file_list)//2-1))
    for idx in range(len(file_list)//2):
        print(idx)
     
        LR_left = cv2.imread(img5_path  + '{:04}_L.jpg'.format(idx+1))
        LR_right = cv2.imread(img5_path  + '{:04}_R.jpg'.format(idx+1))
        HR_left = cv2.imread(img_path_hix  + '{:04}_L.jpg'.format(idx+1))
        HR_right = cv2.imread(img_path_hix  + '{:04}_R.jpg'.format(idx+1))
        print(img_path_hix  + '{:04}_R.jpg'.format(idx+1))
        # print(HR_left.shape)
        # print(HR_right.shape)
        HR_left = HR_left[0:LR_left.shape[0], 0:LR_left.shape[1], :]
        # print(HR_left.shape)
        HR_right = HR_right[0:LR_right.shape[0], 0:LR_right.shape[1], :]
        # print(HR_right.shape)
        LLPIPS.append(calculate_lpips(LR_left,
                    HR_left,
                    0,
                    input_order='HWC'))
        RLPIPS.append(calculate_lpips(LR_right,
                    HR_right,
                    0,
                    input_order='HWC'))
        LPSNR.append(calculate_psnr(LR_left,
                    HR_left,
                    0,
                    input_order='HWC'))
        RPSNR.append(calculate_psnr(LR_right,
                    HR_right,
                    0,
                    input_order='HWC'))
    total=sum(LPSNR)+sum(RPSNR)
    avg=total/(len(LPSNR)+len(RPSNR))
    print(avg)



    total1=sum(LLPIPS)+sum(RLPIPS)
    avg1=total1/(len(LLPIPS)+len(RLPIPS))
    print(avg1)
                    
