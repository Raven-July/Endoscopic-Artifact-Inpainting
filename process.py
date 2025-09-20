import torch
import os
import argparse
import numpy as np
import glob
import cv2
import yaml
from omegaconf import OmegaConf
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from data import dataloaders_duck as dataloaders
from models import duck_net
from stabledelight import YosoDelightPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1
    )
    return mask


def process_with_sd(input_image, args, pipe):
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    # 创建单通道遮罩
    thresh = args.mix_thresh
    mask_single_channel = np.zeros_like(gray_img)  # 创建与灰度图像大小相同的全零数组
    mask_single_channel[gray_img > thresh] = 255  # 将高亮度区域的像素值设置为255
    # 转换为只包含0和255的numpy数组
    mask = (mask_single_channel > 0).astype(np.uint8) * 255
    mask = dilate_mask(mask, args.mix_dilate_size)
    # 对原图像进行掩膜操作，获取遮罩
    _, overexposure_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # 计算消光图
    pipe_out = pipe(
        input_image,
        match_input_resolution=False,
        processing_resolution=args.resolution,
    )
    processed_frame = (pipe_out.prediction.clip(-1, 1) + 1) / 2
    processed_frame = (processed_frame[0] * 255).astype(np.uint8)
    delighted_part = cv2.bitwise_and(
        processed_frame, processed_frame, mask=overexposure_mask
    )
    delighted_part = cv2.GaussianBlur(
        delighted_part,
        (args.mix_blur_size, args.mix_blur_size),
        args.mix_blur_sigma,
        args.mix_blur_sigma,
    )
    _, new_mask = cv2.threshold(
        cv2.cvtColor(delighted_part, cv2.COLOR_RGB2GRAY),
        0,
        255,
        cv2.THRESH_BINARY,
    )
    # 动态图片混合
    # 计算权重
    weight = cv2.cvtColor(delighted_part, cv2.COLOR_RGB2GRAY)
    max = delighted_part.max()
    min = delighted_part.min()
    # print(max, min)
    weight = (weight - min) / (max - min)
    # print(weight.max(), weight.min())
    weight = weight[..., np.newaxis]

    # 创建输出图像
    output = input_image.copy()
    mask_non_zero = new_mask > 0
    output[mask_non_zero] = (
        input_image[mask_non_zero] * (1 - weight[mask_non_zero])
        + processed_frame[mask_non_zero] * weight[mask_non_zero]
    ).astype(np.uint8)

    # 保存图片
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img_path = args.input + "/*"
    input_paths = sorted(glob.glob(img_path))

    output_dataloader = dataloaders.get_dataloaders_output(input_paths)
    # step0: Load models
    print("loading models...")

    # 加载DuckNet模型
    duck_model = duck_net.DuckNet(
        in_channels=3,
        out_channels=1,
        depth=5,
        init_features=16,
        normalization="batch",
        interpolation="nearest",
        out_activation=None,
        use_multiplier=True,
    )
    state_dict = torch.load(args.duck_path)
    duck_model.load_state_dict(state_dict["model_state_dict"])
    duck_model.to(device)
    print("DuckNet model loaded")

    # 加载lama模型
    predict_config = OmegaConf.load(args.lama_config)
    predict_config.model.path = args.lama_path
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    train_config_path = os.path.join(predict_config.model.path, "config.yaml")

    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    checkpoint_path = os.path.join(
        predict_config.model.path, "models", predict_config.model.checkpoint
    )
    lama_model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location="cpu"
    )
    lama_model.freeze()
    if not predict_config.get("refine", False):
        lama_model.to(device)
    print("lama model loaded")

    # 加载StableDelight模型
    if not args.disable_sd:
        pipe = YosoDelightPipeline.from_pretrained(
            args.sd_path,
            trust_remote_code=True,
            variant="fp16",
            torch_dtype=torch.float16,
            t_start=0,
        ).to(device)
        print("StableDelight model loaded")
        return device, output_dataloader, duck_model, lama_model, pipe, input_paths
    else:
        return device, output_dataloader, duck_model, lama_model, input_paths


@torch.no_grad()
def predict(args):
    if args.disable_sd:
        device, output_dataloader, duck_model, lama_model, paths = build(args)
    else:
        device, output_dataloader, duck_model, lama_model, pipe, paths = build(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    duck_model.eval()

    for i, input_img in enumerate(output_dataloader):
        print(
            "Progess:{}/{} Processing image: {}".format(
                i + 1, len(output_dataloader), os.path.basename(paths[i])
            )
        )
        input_img = input_img.to(device)
        # Step_1: Process with DuckNet
        # check image_size
        change = False
        w, h = input_img.shape[2], input_img.shape[3]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        if pad_w != 0 or pad_h != 0:
            resized_img = F.interpolate(
                input_img,
                size=(w + pad_w, h + pad_h),
                mode="bilinear",
                align_corners=False,
            )
            change = True
            output = duck_model(resized_img)
        # predict
        else:
            output = duck_model(input_img)
        # same_size
        if change == True:
            output = F.interpolate(
                output,
                size=(w, h),
                mode="bilinear",
                align_corners=False,
            )
        specular_mask = np.array(output.cpu())
        specular_mask = np.squeeze(specular_mask)
        specular_mask = specular_mask > 0
        specular_mask = (specular_mask * 255).astype(np.uint8)
        if args.trad_help == True:
            # input_img中像素大于230的部分，也在final_mask中标记为255
            img_np = input_img[0].permute(1, 2, 0)
            img_np = img_np.detach().cpu().numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype("uint8")
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            specular_mask[gray > 230] = 255

        # step2: Process with Big-Lama
        if args.lama_dilate_size is not None:
            mask = dilate_mask(specular_mask, args.lama_dilate_size)

        mask = torch.from_numpy(mask).float()
        batch = {}
        # 转换归一化后的图像范围，从[-1,1]换到[0,1]。因为两个网络使用的归一化方法不同
        batch["image"] = (input_img + 1) / 2
        batch["mask"] = mask[None, None]
        unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
        # check image_size
        batch["image"] = pad_tensor_to_modulo(batch["image"], 8)
        batch["mask"] = pad_tensor_to_modulo(batch["mask"], 8)
        batch = move_to_device(batch, device)
        batch["mask"] = (batch["mask"] > 0) * 1
        # predict
        batch = lama_model(batch)
        predict_config = OmegaConf.load(args.lama_config)
        inpainted_img = batch[predict_config.out_key][0].permute(1, 2, 0)
        inpainted_img = inpainted_img.detach().cpu().numpy()
        # same_size
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            inpainted_img = inpainted_img[:orig_height, :orig_width]
        inpainted_img = np.clip(inpainted_img * 255, 0, 255).astype("uint8")

        # step3: Process with StableDelight
        if args.disable_sd:
            output = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR)
        else:
            output = process_with_sd(inpainted_img, args, pipe)
        cv2.imwrite(
            "{}/{}".format(
                args.output,
                os.path.basename(paths[i][:-4] + ".png"),
                # os.path.basename(paths[i]),
            ),  # 处理teeth记得换成png
            output,
        )


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./Test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
    )
    parser.add_argument(
        "--duck_path",
        type=str,
        default="/root/autodl-tmp/pretrained_models/Duck16mod_Spec_99_new.pt",
        help="The path to the Ducknet checkpoint.",
    )
    parser.add_argument(
        "--sd_path",
        type=str,
        required=True,
        help="The path to the StableDelight checkpoint.",
    )
    parser.add_argument(
        "--lama_path",
        type=str,
        default="/root/autodl-tmp/pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--lama_config",
        type=str,
        default="./configs/lama_predict_default.yaml",
        help="The path to the config file of lama model. "
        "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_dilate_size",
        type=int,
        default=12,
        help="Dilate kernel size used in big-lama inpainting process",
    )
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument(
        "--mix_thresh",
        type=int,
        default=190,
        help="Mix_thresh decides which part of the image needs StableDelight for refinement",
    )
    parser.add_argument("--mix_dilate_size", type=int, default=15)
    parser.add_argument("--mix_blur_size", type=int, default=55)
    parser.add_argument("--mix_blur_sigma", type=int, default=10)
    parser.add_argument("--disable_sd", type=bool, default=False)
    parser.add_argument("--trad_help", type=bool, default=False)
    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
