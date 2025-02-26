'''
Training and testing script for CompenNeSt++ (journal extension of cvpr'19 and iccv'19 papers)

This script trains/tests CompenNeSt++ on different dataset specified in 'data_list' below.
The detailed training options are given in 'train_option' below.

1. We start by setting the training environment to GPU (if any).
2. K=20 setups are listed in 'data_list', which are our full compensation benchmark.
3. We set number of training images to 500 and loss function to l1+ssim, you can add other num_train and loss to 'num_train_list' and 'loss_list' for
comparison. Other training options are specified in 'train_option'.
4. The training data 'train_data' and validation data 'valid_data', are loaded in RAM using function 'loadData', and then we train the model with
function 'trainCompenNeStModel'. The training and validation results are both updated in Visdom window (`http://server:8098`) and console.
5. Once the training is finished, we can compensate the desired image. The compensation images 'prj_cmp_test' can then be projected to the surface.

Example:
    python train_compenNeSt++.py

See Models.py for CompenNeSt++ structure.
See trainNetwork.py for detailed training process.
See utils.py for helper functions.

Citation:
    @article{huang2020CompenNeSt++,
        title={End-to-end Full Projector Compensation},
        author={Bingyao Huang and Tao Sun and Haibin Ling},
        year={2021},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)} }

    @inproceedings{huang2019compennet++,
        author = {Huang, Bingyao and Ling, Haibin},
        title = {CompenNet++: End-to-end Full Projector Compensation},
        booktitle = {IEEE International Conference on Computer Vision (ICCV)},
        month = {October},
        year = {2019} }

    @inproceedings{huang2019compennet,
        author = {Huang, Bingyao and Ling, Haibin},
        title = {End-To-End Projector Photometric Compensation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019} }
'''

# %% Set environment
import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

import PIL.Image as Image
from torchvision import transforms

import pytorch_ssim
from src.python.My_PCNet import My_models as models, utils as ut
# import models
from src.python.My_PCNet.img_proc import center_crop as cc
from src.python.My_PCNet.utils import plot_montage
from os.path import join, abspath
from src.python.My_PCNet.utils import print_sys_info, set_torch_reproducibility
from src.python.My_PCNet.My_train_network import get_model_train_cfg, load_mask_info

print_sys_info()

# [reproducibility] did not see significantly differences when set to True, but True is significantly slower.
set_torch_reproducibility(False)

# training configs
data_root = abspath(join(os.getcwd(), '../../../data'))
setup_list = [
    # 'DR',
    'DR2',
    # 'lotion',
    # 'soccer',
    # 'paper_towel',
    # 'volleyball',
    # 'backpack',
    # 'hamper',
    # 'bucket',
    # 'coffee_mug',
    # 'banana',
    # 'book_jacket',
    # 'remote_control',
    # 'mixing_bowl',
    # 'pillow',
]

# pcnet_cfg = get_model_train_cfg(['PCNet'], data_root, setup_list, load_pretrained=True, plot_on=True)
# pcnet_cfg = get_model_train_cfg(['PCNet_no_mask'], data_root, setup_list, load_pretrained=True, plot_on=True)
# pcnet_cfg = get_model_train_cfg(['PCNet_no_rough'], data_root, setup_list, load_pretrained=True, plot_on=True)
# pcnet_cfg = get_model_train_cfg(['PCNet_no_mask_no_rough'], data_root, setup_list, load_pretrained=True, plot_on=True)
# pcnet_cfg = get_model_train_cfg(['My_PCNet'], data_root, setup_list, load_pretrained=True, plot_on=True)
pcnet_cfg = get_model_train_cfg(['My_PCNet_no_mask'], data_root, setup_list, load_pretrained=True, plot_on=True)
# pcnet_cfg = get_model_train_cfg(['My_PCNet_no_rough'], data_root, setup_list, load_pretrained=True, plot_on=True)
# pcnet_cfg = get_model_train_cfg(['My_PCNet_no_mask_no_rough'], data_root, setup_list, load_pretrained=True, plot_on=True)


# stats for different setups
data_root = pcnet_cfg.data_root
# set PyTorch device to GPU
device = torch.device(pcnet_cfg.device)

# log
ret, log_txt_filename, log_xls_filename = ut.init_log_file(join(data_root, '../log'))

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM().cuda()


def tv_loss(image_tensor):
    """
    计算批量图像的 Total Variation Loss，用于约束空间平滑性。

    参数:
        image_tensor (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)。

    返回:
        loss (torch.Tensor): TV Loss 标量。
    """
    # 水平方向变化 (differences between adjacent columns)
    diff_x = image_tensor[:, :, :, 1:] - image_tensor[:, :, :, :-1]
    # 垂直方向变化 (differences between adjacent rows)
    diff_y = image_tensor[:, :, 1:, :] - image_tensor[:, :, :-1, :]

    # 计算 L1 范数，减少图像梯度的变化
    tv_x = torch.abs(diff_x).mean()
    tv_y = torch.abs(diff_y).mean()

    # 总 TV Loss
    loss = tv_x + tv_y
    return loss

# train and evaluate all setups
def test(prj_img, cam_desire, cam_surf):
    setup_path = join(data_root , 'setups', pcnet_cfg.setup_list[0])
    # load training and validation data
    cam_mask, mask_corners, setup_info = load_mask_info(data_root, pcnet_cfg.setup_list[0])
    pcnet_cfg.setup_info = setup_info

    # center crop, decide whether PCNet output is center cropped square image (classifier_crop_sz) or not (cam_im_sz)
    if pcnet_cfg.center_crop:
        cp_sz = setup_info.classifier_crop_sz
        prj_img = cc(prj_img, cp_sz)
        cam_desire = cc(cam_desire, cp_sz)
        cam_surf = cc(cam_surf, cp_sz)

    # surface image for training and validation
    # prj_img = prj_img.to(device)
    # cam_desire = cam_desire.to(device)
    # cam_surf = cam_surf.to(device)


    # stats for different models
    pcnet_cfg.model_name = pcnet_cfg.model_list[0].replace('/', '_')
    pcnet_cfg.setup_name = pcnet_cfg.setup_list[0].replace('/', '_')
    pcnet_cfg.loss = pcnet_cfg.loss_list[0]
    pcnet_cfg.num_train = pcnet_cfg.num_train_list[0]

    for key in ['num_train_list', 'model_list', 'loss_list', 'setup_list']:
        del pcnet_cfg[key]
    # set seed of rng for repeatability
    # ut.reset_rng_seeds(123)

    # create a ShadingNetSPAA model
    shading_net = models.ShadingNetSPAA(use_rough='no_rough' not in pcnet_cfg.model_name)
    if torch.cuda.device_count() >= 1: shading_net = nn.DataParallel(shading_net, device_ids=pcnet_cfg.device_ids).to(device)

    # create a WarpingNet model
    warping_net = models.WarpingNet(out_size=cam_surf.shape[-2:], with_refine='w/o_refine' not in pcnet_cfg.model_name)  # warp prj to cam raw

    # initialize WarpingNet with affine transformation (remember grid_sample is inverse warp, so src is the desired warp
    src_pts    = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
    dst_pts    = np.array(mask_corners[0:3]).astype(np.float32)
    affine_mat = torch.Tensor(cv.getAffineTransform(dst_pts, src_pts))  # prj -> cam
    warping_net.set_affine(affine_mat.flatten())
    if torch.cuda.device_count() >= 1: warping_net = nn.DataParallel(warping_net, device_ids=pcnet_cfg.device_ids).to(device)

    # create a PCNet model using WarpingNet and ShadingNetSPAA
    pcnet = models.PCNet(cam_mask.float(), warping_net, shading_net, fix_shading_net=False, use_mask='no_mask' not in pcnet_cfg.model_name,
                         use_rough='no_rough' not in pcnet_cfg.model_name)
    if torch.cuda.device_count() >= 1: pcnet = nn.DataParallel(pcnet, device_ids=pcnet_cfg.device_ids).to(device)

    print(f'------------------------------------ Loading pretrained {pcnet_cfg.model_name:s} ---------------------------')
    checkpoint_filename = join(data_root, '../checkpoint', ut.opt_to_string(pcnet_cfg) + '.pth')
    print(checkpoint_filename)
    pcnet.load_state_dict(torch.load(checkpoint_filename))

    # [validation phase] after training we evaluate and save results
    print('------------------------------------ Start testing {:s} ---------------------------'.format(pcnet_cfg.model_name))
    torch.cuda.empty_cache()

    # compensate and save images
    ori_desire_test = cam_desire.detach().clone()
    opt_input = prj_img.clone().requires_grad_(True)

    # simplify CompenNet++
    # compen_nest_pp.module.simplify(cam_surf_test[0, ...].unsqueeze(0))

    # compensate using CompenNet++
    pcnet.eval()
    if pcnet_cfg['plot_on']:
        title = ut.opt_to_string(pcnet_cfg)

    vis_valid_fig = None
    pred = pcnet(opt_input, cam_surf)
    pred.data.clamp_(0, 1)
    vis_valid_fig = plot_montage(torch.cat(
        (F.interpolate(cam_surf, size=(256, 256), mode='bilinear', align_corners=False), opt_input, F.interpolate(pred, size=(256, 256), mode='bilinear', align_corners=False)),
        dim=0), win=vis_valid_fig, title='[Valid]' + title)

    # create image save path
    # cmp_folder_name = '{}_{}_{}_{}_{}'.format(train_option['model_name'], loss, num_train, train_option['batch_size'],
    #                                           train_option['max_iters'])
    # prj_cmp_path = fullfile(data_root, 'prj/cmp/test', cmp_folder_name)
    # if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)
    #
    # # save images
    # saveImgs(prj_cmp_test, prj_cmp_path)  # compensated testing images, i.e., to be projected to the surface
    # print('Compensation images saved to ' + prj_cmp_path)

    # clear cache
    del pcnet, warping_net
    torch.cuda.empty_cache()
    print('-------------------------------------- Done! ---------------------------\n')

if __name__ == "__main__":
    transform = transforms.ToTensor()  # 将图片转换为Tensor
    cam_desire = transform(Image.open('../../tep/' + setup_list[0] + '/desire.png').convert("RGB")).to(device).unsqueeze(0)
    cam_surf = transform(Image.open('../../tep/' + setup_list[0] + '/surf.png')).to(device).unsqueeze(0)
    # prj_img = (2 * torch.randn(1, 3, 256, 256).to(device).data.clamp_(-0.5, 0.5) + 1) / 2
    # prj_img = torch.ones(1, 3, 256, 256).to(device) * 0.309
    checkerboard_rgb = torch.zeros(1, 3, 256, 256)
    # 红绿相间：奇数行奇数列为绿色，偶数行偶数列为红色
    # 在 RGB 模式下，红色的通道为 [1, 0, 0]，绿色的通道为 [0, 1, 0]
    checkerboard_rgb[:, 0, 1::2, ::2] = 1  # 红色通道 (1, 0, 0)
    # checkerboard_rgb[:, 1, 1::2, ::2] = 1  # 绿色通道 (0, 1, 0)
    # checkerboard_rgb[:, 2, 1::2, ::2] = 1  # 蓝色通道 (0, 0, 1)
    checkerboard_rgb[:, 0, ::2, 1::2] = 1  # 红色通道 (1, 0, 0)
    # checkerboard_rgb[:, 1, ::2, 1::2] = 1  # 绿色通道 (0, 1, 0)
    # checkerboard_rgb[:, 2, ::2, 1::2] = 1  # 蓝色通道 (0, 0, 1)
    prj_img = checkerboard_rgb.to(device)
    test(prj_img, cam_desire, cam_surf)

