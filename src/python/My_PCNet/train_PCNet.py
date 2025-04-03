import os
from os.path import join, abspath
import argparse

# set which GPU(s) to use
parser = argparse.ArgumentParser(description="示例：Python 脚本参数")
parser.add_argument('--device', nargs='+', type=int, default=[0], help="指定使用的 GPU 设备列表，如 0 1 2")
args = parser.parse_args()
gpu_list = ",".join(map(str, args.device))  # 转换为字符串 "0,1,2"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print(os.environ['CUDA_VISIBLE_DEVICES'])

from src.python.My_PCNet.utils import print_sys_info, set_torch_reproducibility
from src.python.My_PCNet.My_train_network import train_eval_pcnet, get_model_train_cfg
# from train_network import train_eval_pcnet, train_eval_compennet_pp, get_model_train_cfg

print_sys_info()

# [reproducibility] did not see significantly differences when set to True, but True is significantly slower.
set_torch_reproducibility(False)

# training configs
data_root = abspath(join(os.getcwd(), '../../../data'))
setup_list = [
    # 'DR',
    # 'DR2',
    # 'DR3'
    'pillow1',
    # 'lavander1',
    # 'cloud1',
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


# pcnet_cfg       = get_model_train_cfg(['PCNet'], data_root, setup_list, load_pretrained=False, plot_on=True)
pcnet_cfg       = get_model_train_cfg(['My_PCNet_no_mask'], data_root, setup_list, center_crop=True, load_pretrained=False, plot_on=True)
_, pcnet_ret, _ = train_eval_pcnet(pcnet_cfg)