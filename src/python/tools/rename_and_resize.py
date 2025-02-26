from PIL import Image

from src.python.My_PCNet.My_models import ShadingNetSPAA


def resize_images(folder_path, target_size=(320, 240)):
    """
    Recursively resize all images in the given folder to the specified size.

    Parameters:
        folder_path (str): Path to the folder containing images.
        target_size (tuple): Target size as (width, height).
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    # Resize the image and save it back to the same path
                    img_resized = img.resize(target_size, Image.ANTIALIAS)
                    img_resized.save(file_path)
                    print(f"Resized: {file_path}")
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

import os


def rename_images(folder_path):
    # 获取文件夹下的所有文件
    files = os.listdir(folder_path)

    # 过滤出图片文件（例如 .png, .jpg 等格式）
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # 按文件名排序（确保顺序一致）
    image_files.sort()

    # 遍历图片文件并重命名
    for idx, file_name in enumerate(image_files, start=1):
        # 构造新文件名
        new_name = f"img_{idx:04d}.png"

        # 获取文件的完整路径
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {file_name} -> {new_name}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    # folder_path = 'noise'
    # if os.path.isdir(folder_path):
    #     rename_images(folder_path)
    # else:
    #     print("The specified path is not a valid directory.")
    original_model = ShadingNetSPAA()
    # modified_model = ShadingNetSPAA_Attention()

    original_params = count_parameters(original_model)
    # modified_params = count_parameters(modified_model)

    print("Original Model Parameters:", original_params)
    # print("Modified Model Parameters:", modified_params)

