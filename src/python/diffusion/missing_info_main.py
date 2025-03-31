import logging
import os
import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from PIL import Image

from src.python.diffusion.datasets.utils import normalize
from src.python.diffusion.datasets import load_lama_celebahq, load_imagenet, load_test
from guided_diffusion import (
    DDIMSampler,
    O_DDIMSampler,
    DDNMSampler,
    DDRMSampler,
    DPSSampler,
    Info_O_DDIMSampler,
    Test_DDIMSampler
)
from guided_diffusion import dist_util
from guided_diffusion.ddim import R_DDIMSampler
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    select_args,
    create_classifier,
    classifier_defaults,
)
from metrics import LPIPS, PSNR, SSIM, Metric
from utils import save_grid, save_image, normalize_image
from utils.config import Config
from utils.logger import get_logger, logging_info
from utils.nn_utils import get_all_paths, set_random_seed
from utils.result_recorder import ResultRecorder
from utils.timer import Timer


def prepare_model(algorithm, conf, device):
    logging_info("Prepare model...")
    unet = create_model(**select_args(conf, model_defaults().keys()), conf=conf)
    SAMPLER_CLS = {
        "repaint": SpacedDiffusion,
        "ddim": DDIMSampler,
        "o_ddim": O_DDIMSampler,
        "resample": R_DDIMSampler,
        "ddnm": DDNMSampler,
        "ddrm": DDRMSampler,
        "dps": DPSSampler,
        "info_o_ddim": Info_O_DDIMSampler,
        "test": Test_DDIMSampler,
    }
    sampler_cls = SAMPLER_CLS[algorithm]
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )

    logging_info(f"Loading model from {conf.model_path}...")
    unet.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path), map_location="cpu"
        ), strict=False
    )
    unet.to(device)
    if conf.use_fp16:
        unet.convert_to_fp16()
    unet.eval()
    return unet, sampler


def prepare_classifier(conf, device):
    logging_info("Prepare classifier...")
    classifier = create_classifier(
        **select_args(conf, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.classifier_path), map_location="cpu"
        )
    )
    classifier.to(device)
    classifier.eval()
    return classifier


def prepare_data(
        dataset_name, mask_type="half", dataset_starting_index=-1, dataset_ending_index=-1, maxlen=100, split='DR2'
):
    if dataset_name == "celebahq":
        datas = load_lama_celebahq(mask_type=mask_type)
    elif dataset_name == "imagenet":
        datas = load_imagenet(mask_type=mask_type, max_len=maxlen, split=split)
    elif dataset_name == "imagenet64":
        datas = load_imagenet(mask_type=mask_type, shape=(64, 64))
    elif dataset_name == "imagenet128":
        datas = load_imagenet(mask_type=mask_type, shape=(128, 128))
    elif dataset_name == "imagenet512":
        datas = load_imagenet(mask_type=mask_type, shape=(512, 512))
    elif dataset_name == "celeb_test":
        datas = load_test(mask_type=mask_type)
    else:
        raise NotImplementedError

    dataset_starting_index = (
        0 if dataset_starting_index == -1 else dataset_starting_index
    )
    dataset_ending_index = (
        len(datas) if dataset_ending_index == -1 else dataset_ending_index
    )
    datas = datas[dataset_starting_index:dataset_ending_index]

    logging_info(f"Load {len(datas)} samples")
    return datas


def all_exist(paths):
    for p in paths:
        if not os.path.exists(p):
            return False
    return True


def main():
    ###################################################################################
    # prepare config, logger and recorder
    ###################################################################################
    config = Config(default_config_file="src/python/diffusion/configs/imagenet.yaml", use_argparse=True)
    config.show()
    torch.cuda.set_device(config.device)

    all_paths = get_all_paths(config.outdir)
    config.dump(all_paths["path_config"])
    set_random_seed(random.randint(0,100000000), deterministic=False, no_torch=False, no_tf=True)
    ###################################################################################
    # prepare model and device
    ###################################################################################
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    unet, sampler = prepare_model(config.algorithm, config, device)

    def model_fn(x, t, y=None, gt=None, **kwargs):
        return unet(x, t, y if config.class_cond else None, gt=gt)

    cond_fn = None

    METRICS = {
        "lpips": Metric(LPIPS("alex", device)),
        "psnr": Metric(PSNR(), eval_type="max"),
        "ssim": Metric(SSIM(), eval_type="max"),
    }
    for setup in config.setup_list.split(","):
        ###################################################################################
        # prepare data
        ###################################################################################
        if config.input_image == "":  # if input image is not given, load dataset
            datas = prepare_data(
                config.dataset_name,
                config.mask_type,
                config.dataset_starting_index,
                config.dataset_ending_index,
                maxlen=config.num_samples,
                split=setup,
            )
        else:
            # NOTE: the model should accepet this input image size
            image = normalize(Image.open(config.input_image).convert("RGB"))
            if config.mode != "super_resolution":
                mask = (
                    torch.from_numpy(np.array(Image.open(config.mask).convert("1"), dtype=np.float32))
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            else:
                mask = torch.from_numpy(np.array([0]))  # just a dummy value
            datas = [(image, mask, "sample0")]



        ###################################################################################
        # start sampling
        ###################################################################################
        get_logger(all_paths["path_log"], force_add_handler=True)
        recorder = ResultRecorder(
            path_record=all_paths["path_record"],
            initial_record=config,
            use_git=config.use_git,
        )
        logging_info("Start sampling")
        timer, num_image = Timer(), 0
        batch_size = config.n_samples
        counter = 0
        final_loss = []
        for data in tqdm(datas):
            if config.class_cond:
                image, mask, image_name, class_id = data
            else:
                image, mask, image_name, class_id= data
                class_id = None
            # prepare save dir
            outpath = os.path.join(config.outdir, setup, str(config.start_index) + '_' + str(config.interval_index), image_name)
            os.makedirs(outpath, exist_ok=True)
            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)
            base_count = 0
            grid_count = max(len(os.listdir(outpath)) - 3, 0)

            # prepare batch data for processing
            batch = {"image": image.to(device), "mask": mask.to(device)}
            model_kwargs = {
                "image_name":image_name,
                "gt": batch["image"].repeat(batch_size, 1, 1, 1),
                "gt_keep_mask": batch["mask"].repeat(batch_size, 1, 1, 1),
                "outdir": config.outdir,
                "start_index": config.start_index,
                "interval_index": config.interval_index,
            }
            if config.missing_info:
                mask = model_kwargs["gt_keep_mask"]
                mask = (mask * 255).to(torch.uint8)
                mask = mask.squeeze().cpu().numpy()
                # 面积
                White = 255
                white_pixel_count = np.sum(mask == 255 if White == 255 else mask == 0)
                black_pixel_count = np.sum(mask == 0 if White == 255 else mask == 255)
                # 找到黑色像素的位置
                black_pixels = np.argwhere(mask == 0 if White == 255 else mask == 255)
                # 找到白色像素的位置
                white_pixels = np.argwhere(mask == 255 if White == 255 else mask == 0)
                # 使用sklearn的NearestNeighbors找到每个黑色像素十个最近邻白色像素
                n_neighbors = 10
                power = 1
                clip = 7
                nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(white_pixels)
                distances, indices = nbrs.kneighbors(black_pixels)
                # # 计算每个黑色像素的十个最近邻白色像素的欧式距离之和
                # total_distance = np.sum(np.sum(distances, axis=1))/255/255/100
                # 计算每个黑色像素的十个最近邻白色像素的欧式距离的倒数的平方
                inverse_distances_squared = 1 / (distances ** power)
                # print(inverse_distances_squared)
                a = np.clip(np.sum(inverse_distances_squared, axis=1), 0, clip)
                total_inverse_distance_squared = np.sum(a)
                lost_info = (clip * black_pixel_count - total_inverse_distance_squared) / 255 / 255 / clip * 100
                float_mask = np.zeros_like(mask, dtype=np.float32)
                float_mask[black_pixels[:, 0], black_pixels[:, 1]] = a / clip
                float_mask = torch.from_numpy(float_mask).to(device).repeat(1,1,1,1)
                model_kwargs["weight_mask_known"] = float_mask

                model_kwargs["lost_info"] = lost_info * 0.01
                model_kwargs["known_info"] = 1 - lost_info * 0.01

                #计算mask的missing information
                White = 0
                white_pixel_count = np.sum(mask == 255 if White == 255 else mask == 0)
                black_pixel_count = np.sum(mask == 0 if White == 255 else mask == 255)
                # 找到黑色像素的位置
                black_pixels = np.argwhere(mask == 0 if White == 255 else mask == 255)
                # 找到白色像素的位置
                white_pixels = np.argwhere(mask == 255 if White == 255 else mask == 0)
                # 使用sklearn的NearestNeighbors找到每个黑色像素十个最近邻白色像素
                n_neighbors = 10
                power = 1
                clip = 7
                nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(white_pixels)
                distances, indices = nbrs.kneighbors(black_pixels)
                # # 计算每个黑色像素的十个最近邻白色像素的欧式距离之和
                # total_distance = np.sum(np.sum(distances, axis=1))/255/255/100
                # 计算每个黑色像素的十个最近邻白色像素的欧式距离的倒数的平方
                inverse_distances_squared = 1 / (distances ** power)
                # print(inverse_distances_squared)
                a = np.clip(np.sum(inverse_distances_squared, axis=1), 0, clip)
                total_inverse_distance_squared = np.sum(a)
                lost_info = (clip * black_pixel_count - total_inverse_distance_squared) / 255 / 255 / clip * 100
                float_mask = np.zeros_like(mask, dtype=np.float32)
                float_mask[black_pixels[:, 0], black_pixels[:, 1]] = a / clip
                float_mask *= 256 * 256 / float_mask.sum()
                float_mask = torch.from_numpy(float_mask).to(device).repeat(1, 1, 1, 1)
                model_kwargs["weight_mask_unknown"] = float_mask
                print(lost_info)
            if config.class_cond:
                if config.cond_y is not None:
                    classes = torch.ones(batch_size, dtype=torch.long, device=device)
                    model_kwargs["y"] = classes * config.cond_y
                elif config.classifier_path is not None:
                    classes = torch.full((batch_size,), class_id, device=device)
                    model_kwargs["y"] = classes

            shape = (batch_size, 3, config.image_size, config.image_size)

            all_metric_paths = [
                os.path.join(outpath, i + ".last")
                for i in (list(METRICS.keys()) + ["final_loss"])
            ]
            if config.get("resume", False) and all_exist(all_metric_paths):
                for metric_name, metric in METRICS.items():
                    metric.dataset_scores += torch.load(
                        os.path.join(outpath, metric_name + ".last")
                    )
                logging_info("Results exists. Skip!")
            else:
                # sample images
                samples = []
                for n in range(config.n_iter):
                    timer.start()
                    result = sampler.p_sample_loop(
                        model_fn,
                        shape=shape,
                        model_kwargs=model_kwargs,
                        cond_fn=cond_fn,
                        device=device,
                        progress=True,
                        return_all=True,
                        conf=config,
                        file_number=n,
                        setup=setup,
                        sample_dir=outpath if config["debug"] else None,
                    )
                    timer.end()

                    for metric in METRICS.values():
                        metric.update(result["sample"], batch["image"])

                    if "loss" in result.keys() and result["loss"] is not None:
                        recorder.add_with_logging(
                            key=f"loss_{image_name}_{n}", value=result["loss"]
                        )
                        final_loss.append(result["loss"])
                    else:
                        final_loss.append(None)

                    inpainted = normalize_image(result["sample"])
                    samples.append(inpainted.detach().cpu())

                samples = torch.cat(samples)
                summary_path = os.path.join(config.outdir, setup, str(config.start_index) + '_' + str(config.interval_index), "summary")
                os.makedirs(summary_path, exist_ok=True)
                save_grid(samples, os.path.join(summary_path, image_name + '.png'), nrow=batch_size,)
                # save images
                # save gt images
                save_grid(normalize_image(batch["image"]), os.path.join(outpath, f"gt.png"))
                save_grid(
                    normalize_image(batch["image"] * batch["mask"]),
                    os.path.join(outpath, f"masked.png"),
                )
                # save generations
                for sample in samples:
                    save_image(sample, os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1
                save_grid(
                    samples,
                    os.path.join(outpath, f"grid-{grid_count:04}.png"),
                    nrow=batch_size,
                )
                # save metrics
                for metric_name, metric in METRICS.items():
                    torch.save(metric.dataset_scores[-config.n_iter:], os.path.join(outpath, metric_name + ".last"))

                torch.save(
                    final_loss[-config.n_iter:], os.path.join(outpath, "final_loss.last"))

                num_image += 1
                last_duration = timer.get_last_duration()
                logging_info(
                    "It takes %.3lf seconds for image %s"
                    % (float(last_duration), image_name)
                )

            # report batch scores
            for metric_name, metric in METRICS.items():
                recorder.add_with_logging(
                    key=f"{metric_name}_score_{image_name}",
                    value=metric.report_batch(),
                )

        # report over all results
        for metric_name, metric in METRICS.items():
            mean, colbest_mean = metric.report_all()
            recorder.add_with_logging(key=f"mean_{metric_name}", value=mean)
            recorder.add_with_logging(
                key=f"best_mean_{metric_name}", value=colbest_mean)
        if len(final_loss) > 0 and final_loss[0] is not None:
            recorder.add_with_logging(
                key="final_loss",
                value=np.mean(final_loss),
            )
        if num_image > 0:
            recorder.add_with_logging(
                key="mean time", value=timer.get_cumulative_duration() / num_image
            )

        logging_info(
            f"Your samples are ready and waiting for you here: \n{config.outdir} \n"
            f" \nEnjoy."
        )
        recorder.end_recording()


if __name__ == "__main__":
    main()
