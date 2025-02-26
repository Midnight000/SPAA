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
import pytorch_ssim
import models
from img_proc import center_crop as cc
from src.python.My_PCNet.utils import vis, plot_montage, append_data_point
from src.python.My_PCNet import utils as ut
from os.path import join, abspath
from src.python.My_PCNet.utils import print_sys_info, set_torch_reproducibility
from My_train_network import get_model_train_cfg, load_data

print_sys_info()

# [reproducibility] did not see significantly differences when set to True, but True is significantly slower.
set_torch_reproducibility(False)

# training configs
data_root = abspath(join(os.getcwd(), '../../../data'))
setup_list = [
    'DR',
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

pcnet_cfg = get_model_train_cfg(['PCNet', 'PCNet_no_rough'], data_root, setup_list, load_pretrained=True, plot_on=True)

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
for setup_name in pcnet_cfg.setup_list:
    setup_path = join(data_root , 'setups', setup_name)
    # load training and validation data
    cam_scene, cam_train, cam_valid, prj_train, prj_valid, cam_mask, mask_corners, setup_info = load_data(data_root, setup_name)
    pcnet_cfg.setup_info = setup_info

    # center crop, decide whether PCNet output is center cropped square image (classifier_crop_sz) or not (cam_im_sz)
    if pcnet_cfg.center_crop:
        cp_sz = setup_info.classifier_crop_sz
        cam_scene = cc(cam_scene, cp_sz)
        cam_train = cc(cam_train, cp_sz)
        cam_valid = cc(cam_valid, cp_sz)
        cam_mask  = cc(cam_mask , cp_sz)

    # surface image for training and validation
    cam_scene = cam_scene.to(device)
    cam_scene_train = cam_scene
    cam_scene_valid = cam_scene.expand(cam_valid.shape[0], -1, -1, -1)

    # convert valid data to CUDA tensor if you have sufficient GPU memory (significant speedup)
    cam_valid = cam_valid.to(device)
    prj_valid = prj_valid.to(device)

    # validation data, 200 image pairs
    valid_data = dict(cam_scene=cam_scene_valid, cam_valid=cam_valid, prj_valid=prj_valid)

    # stats for different #Train
    for num_train in pcnet_cfg.num_train_list:
        cfg = pcnet_cfg.copy()
        cfg.num_train = num_train
        for key in ['num_train_list', 'model_list', 'loss_list', 'setup_list']:
            del cfg[key]

        # select a subset to train
        train_data = dict(cam_scene=cam_scene_train, cam_train=cam_train[:num_train], prj_train=prj_train[:num_train], mask=cam_mask)

        # stats for different models
        for model_name in pcnet_cfg.model_list:
            cfg.model_name = model_name.replace('/', '_')

            # stats for different loss functions
            for loss in pcnet_cfg.loss_list:

                # train option for current configuration, i.e., setup name and loss function
                cfg.setup_name = setup_name.replace('/', '_')
                cfg.loss = loss
                model_version = f'{cfg.model_name}_{loss}_{num_train}_{cfg.batch_size}_{cfg.max_iters}'

                # set seed of rng for repeatability
                ut.reset_rng_seeds(123)

                # create a ShadingNetSPAA model
                shading_net = models.ShadingNetSPAA(use_rough='no_rough' not in model_name)
                if torch.cuda.device_count() >= 1: shading_net = nn.DataParallel(shading_net, device_ids=cfg.device_ids).to(device)

                # create a WarpingNet model
                warping_net = models.WarpingNet(out_size=cam_train.shape[-2:], with_refine='w/o_refine' not in model_name)  # warp prj to cam raw

                # initialize WarpingNet with affine transformation (remember grid_sample is inverse warp, so src is the desired warp
                src_pts    = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
                dst_pts    = np.array(mask_corners[0:3]).astype(np.float32)
                affine_mat = torch.Tensor(cv.getAffineTransform(dst_pts, src_pts))  # prj -> cam
                warping_net.set_affine(affine_mat.flatten())
                if torch.cuda.device_count() >= 1: warping_net = nn.DataParallel(warping_net, device_ids=cfg.device_ids).to(device)

                # create a PCNet model using WarpingNet and ShadingNetSPAA
                pcnet = models.PCNet(cam_mask.float(), warping_net, shading_net, fix_shading_net=False, use_mask='no_mask' not in model_name,
                                     use_rough='no_rough' not in model_name)
                if torch.cuda.device_count() >= 1: pcnet = nn.DataParallel(pcnet, device_ids=cfg.device_ids).to(device)

                print(f'------------------------------------ Loading pretrained {model_name:s} ---------------------------')
                checkpoint_filename = join(data_root, '../checkpoint', ut.opt_to_string(cfg) + '.pth')
                pcnet.load_state_dict(torch.load(checkpoint_filename))

                # [validation phase] after training we evaluate and save results
                print('------------------------------------ Start testing {:s} ---------------------------'.format(model_name))
                torch.cuda.empty_cache()

                # desired test images are created such that they can fill the optimal displayable area (see paper for detail)
                desire_test_path = join(setup_path, 'cam/desire/inpaint')
                assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)

                # compensate and save images
                ori_desire_test = ut.readImgsMT(desire_test_path).to(device)[0:20]
                # desire_test = ori_desire_test.clone()
                cam_scene_test = cam_scene.expand_as(ori_desire_test).to(device)[:20]
                opt_input = torch.ones((cam_scene_test.shape[0], prj_valid.shape[1], prj_valid.shape[2], prj_valid.shape[3]), device='cuda')
                opt_input = (opt_input * 0.3).requires_grad_(True)

                # simplify CompenNet++
                # compen_nest_pp.module.simplify(cam_surf_test[0, ...].unsqueeze(0))

                # compensate using CompenNet++
                lr_xt = 1e4
                pcnet.eval()
                if cfg['plot_on']:
                    title = ut.opt_to_string(cfg)
                    # intialize visdom figures
                    vis_curve_fig = vis.line(X=np.array([0]), Y=np.array([0]), name='origin',
                                             opts=dict(width=1300, height=500, markers=True, markersize=3,
                                                       layoutopts=dict(
                                                           plotly=dict(title={'text': title, 'font': {'size': 24}},
                                                                       font={'family': 'Arial', 'size': 20},
                                                                       hoverlabel={'font': {'size': 20}},
                                                                       xaxis={'title': 'Iteration'},
                                                                       yaxis={'title': 'Metrics',
                                                                              'hoverformat': '.4f'}))))
                iters = 0
                lambda_l1 = 1
                lambda_ssim = 1
                lambda_tv = 1.5
                optimizer = torch.optim.Adam([opt_input], lr=3e-2)
                # omega = 0

                vis_valid_fig = None
                vis_train_fig = None
                for i in range(200):
                    iters += 1
                    if iters % 50 == 0:
                        lambda_tv *= 1.3

                    pred = pcnet(opt_input, cam_scene_test)
                    loss_l1 = l1_fun(pred, ori_desire_test)
                    loss_ssim = 1 * (1 - ssim_fun(pred, ori_desire_test))
                    loss_tv = tv_loss(opt_input)
                    loss_diff = lambda_l1 * loss_l1 + lambda_ssim * loss_ssim + lambda_tv * loss_tv
                    # desire_test_grad_loss_diff = torch.autograd.grad(
                    #     loss_diff, opt_input, retain_graph=False, create_graph=False
                    # )[0].detach()


                    optimizer.zero_grad()
                    loss_diff.backward()
                    idx = np.array([1, 2, 3, 18, 20]) - 1  # fix validatio visulization
                    vis_valid_fig = plot_montage( ori_desire_test[idx], opt_input[idx], pred[idx],
                                                win=vis_valid_fig, title='[Valid]' + title)
                    append_data_point(iters, loss_diff.detach().item(), vis_curve_fig,
                                    'loss_compennest++')
                    optimizer.step()
                    # opt_input = opt_input - desire_test_grad_loss_diff * lr_xt
                    opt_input.data.clamp_(0, 1)

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

# average all setups' metrics and save to log
for model_name in pcnet_cfg.model_list:
    ret.loc[len(ret)] = ret.loc[ret['Model'] == model_name].mean(axis=0, numeric_only=True)
    ret.loc[len(ret) - 1, ['Setup', 'Model']] = [f'[mean]_{len(pcnet_cfg.setup_list)}_setups', model_name]

# ret.loc[len(ret)] = ret.mean(axis=0, numeric_only=True)
# ret.loc[len(ret) - 1, 'Setup'] = '[mean]'
print(ret.to_string(justify='center', float_format='%.4f'))
print('-------------------------------------- End of result table ---------------------------\n')
ut.write_log_file(ret, log_txt_filename, log_xls_filename)  # log of all setups
