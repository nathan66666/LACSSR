import cv2
import math
import numpy as np
import lpips
import torch

from basicsr.metrics.metric_util import reorder_image,to_y_channel
from basicsr.data.transforms import totensor

# loss_fn_alex = None



# def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, strict_shape=True, **kwargs):
#     """Calculate LPIPS

#     Ref: https://github.com/richzhang/PerceptualSimilarity

#     Args:
#         img (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#         crop_border (int): Cropped pixels in each edge of an image. These
#             pixels are not involved in the PSNR calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'.
#             Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

#     Returns:
#         float: psnr result.
#     """
#     #print('lpips',img.shape , img2.shape)
#     global loss_fn_alex
#     if loss_fn_alex is None:
#         loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores

#         if torch.cuda.is_available():
#             loss_fn_alex.cuda()
#     if strict_shape:
#         assert img.shape == img2.shape, (f'Image shapes are differnet: {img.shape}, {img2.shape}.')
#     else:
#         h, w, c = img.shape
#         img2 = img2[0:h, 0:w, 0:c]
#         if img.shape != img2.shape:
#             h, w, c = img2.shape
#             img = img[0:h, 0:w, 0:c]
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
#     img = reorder_image(img, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)

#     if crop_border != 0:
#         img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

#     if test_y_channel:
#         img = to_y_channel(img)
#         img2 = to_y_channel(img2)

#     def np2tensor(x):
#         """

#         Args:
#             x: RGB [0 ~ 255] HWC ndarray

#         Returns: RGB [-1, 1]

#         """
#         print(x.shape)
#         return torch.tensor((x * 2 / 255.0) - 0.5).permute(2, 0, 1).unsqueeze(0).float()

#     # np2tensor
#     img = np2tensor(img)
#     img2 = np2tensor(img2)

#     if torch.cuda.is_available():
#         img = img.cuda()
#         img2 = img2.cuda()

#     with torch.no_grad():
#         print(img.shape,img2.shape)
#         d = loss_fn_alex(img, img2)
#     return d.view(1).cpu().numpy()[0]


def calculate_lpips(img,
                    img2,
                    crop_border,
                    input_order='HWC',
                    test_y_channel=False):
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

    assert img.shape == img2.shape, (
        f'Image shapes are differnet: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        img = img.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)


    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float32)
    img2 = img2.astype(np.float32)

    img, img2 = totensor([img, img2], bgr2rgb=False, float32=True)

    img = img.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    img = (img / 255. - 0.5) * 2
    img2 = (img2 / 255. - 0.5) * 2

    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False) # best forward scores
    #print(img.shape)
    img = img.chunk(2, dim=1)
    img2 = img2.chunk(2, dim=1)
    i=0
    metric = 0.
    for x1 in img:
        #print(x1.shape,img2[i].shape)
        #print(metric)
        metric += loss_fn_alex(x1, img2[i]).squeeze(0).float().detach().cpu().numpy()
        i+=1
        #print(metric)
        
        #print(metric.mean())
        return metric.mean()
def calculate_lpips_sigle(img,
                    img2,
                    crop_border,
                    input_order='HWC',
                    test_y_channel=False):
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

    assert img.shape == img2.shape, (
        f'Image shapes are differnet: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)


    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float32)
    img2 = img2.astype(np.float32)

    img, img2 = totensor([img, img2], bgr2rgb=False, float32=True)

    img = img.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    img = (img / 255. - 0.5) * 2
    img2 = (img2 / 255. - 0.5) * 2

    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False) # best forward scores

    metric = loss_fn_alex(img, img2).squeeze(0).float().detach().cpu().numpy()
    return metric.mean()


def calculate_lpips_left(img,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    assert input_order == 'HWC'
    assert crop_border == 0

    img = img[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_lpips_sigle(img=img, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel)
