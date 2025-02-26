import os
from os.path import join, abspath

# set which GPU(s) to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import print_sys_info, set_torch_reproducibility
# from My_train_network import train_eval_pcnet, train_eval_compennet_pp, get_model_train_cfg
from train_network import train_eval_pcnet, train_eval_compennet_pp, get_model_train_cfg
from projector_based_attack import summarize_all_attackers

print_sys_info()

# [reproducibility] did not see significantly differences when set to True, but True is significantly slower.
set_torch_reproducibility(False)

# training configs
data_root = abspath(join(os.getcwd(), '../../../data'))
setup_list = [
    # 'lotion',
    # 'lotion',
    # 'lotion',
    # 'lotion',
    # 'lotion',
    # 'soccer',
    # 'soccer',
    # 'soccer',
    # 'soccer',
    # 'soccer',
    # 'paper_towel',
    # 'paper_towel',
    # 'paper_towel',
    # 'paper_towel',
    # 'paper_towel',
    # 'volleyball',
    # 'backpack',
    # 'backpack',
    # 'backpack',
    # 'backpack',
    # 'backpack',
    # 'hamper',
    # 'hamper',
    # 'hamper',
    # 'hamper',
    # 'hamper',
    'bucket',
    'bucket',
    'bucket',
    'bucket',
    'bucket',
    'bucket',
    'bucket',
    'bucket',
    'bucket',
    'bucket',
    # 'coffee_mug',
    # 'coffee_mug',
    # 'coffee_mug',
    # 'coffee_mug',
    # 'coffee_mug',
    'banana',
    'banana',
    'banana',
    'banana',
    'banana',
    'banana',
    'banana',
    'banana',
    'banana',
    'banana',
    # 'book_jacket',
    # 'book_jacket',
    # 'book_jacket',
    # 'book_jacket',
    # 'book_jacket',
    # 'remote_control',
    # 'remote_control',
    # 'remote_control',
    # 'remote_control',
    # 'remote_control',
    # 'mixing_bowl',
    # 'mixing_bowl',
    # 'mixing_bowl',
    # 'mixing_bowl',
    # 'mixing_bowl',
    # 'pillow',
    # 'pillow',
    # 'pillow',
    # 'pillow',
    # 'pillow',
]

# pcnet_cfg       = get_model_train_cfg(['PCNet'], data_root, setup_list, load_pretrained=False, plot_on=True)
pcnet_cfg       = get_model_train_cfg(['PCNet'], data_root, setup_list, load_pretrained=False, plot_on=True)
_, pcnet_ret, _ = train_eval_pcnet(pcnet_cfg)