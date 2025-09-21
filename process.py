import os
import sys
import glob
import cv2
import yaml
import torch
import argparse
import numpy as np
import logging
from pathlib import Path
from omegaconf import OmegaConf
import torch.nn.functional as F

# Local imports
from data import dataloaders_duck as dataloaders
from models import duck_net
from stabledelight import YosoDelightPipeline

# Extend path for lama modules
sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


# ---------------------- Utility Functions ----------------------
def dilate_mask(mask: np.ndarray, dilate_factor: int = 15) -> np.ndarray:
    """Dilate a binary mask using OpenCV."""
    kernel = np.ones((dilate_factor, dilate_factor), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def process_with_sd(input_image: np.ndarray, args, pipe) -> np.ndarray:
    """Post-process image using StableDelight pipeline."""
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    # Step 1: Create mask for overexposed regions
    mask = (gray_img > args.mix_thresh).astype(np.uint8) * 255
    mask = dilate_mask(mask, args.mix_dilate_size)
    _, overexposure_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Step 2: Run StableDelight
    pipe_out = pipe(
        input_image,
        match_input_resolution=False,
        processing_resolution=args.resolution,
    )
    processed_frame = ((pipe_out.prediction.clip(-1, 1) + 1) / 2)[0]
    processed_frame = (processed_frame * 255).astype(np.uint8)

    # Step 3: Extract & blend delighted region
    delighted_part = cv2.bitwise_and(
        processed_frame, processed_frame, mask=overexposure_mask
    )
    delighted_part = cv2.GaussianBlur(
        delighted_part,
        (args.mix_blur_size, args.mix_blur_size),
        args.mix_blur_sigma,
    )

    _, new_mask = cv2.threshold(
        cv2.cvtColor(delighted_part, cv2.COLOR_RGB2GRAY),
        0,
        255,
        cv2.THRESH_BINARY,
    )

    weight = cv2.cvtColor(delighted_part, cv2.COLOR_RGB2GRAY).astype(np.float32)
    weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
    weight = weight[..., np.newaxis]

    output = input_image.copy()
    mask_non_zero = new_mask > 0
    output[mask_non_zero] = (
        input_image[mask_non_zero] * (1 - weight[mask_non_zero])
        + processed_frame[mask_non_zero] * weight[mask_non_zero]
    ).astype(np.uint8)

    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


# ---------------------- Model Building ----------------------
def build_models(args):
    """Load DuckNet, Lama, and optionally StableDelight models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DuckNet
    print("Loading DuckNet...")
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
    state_dict = torch.load(args.duck_path, map_location=device)
    duck_model.load_state_dict(state_dict["model_state_dict"])
    duck_model.to(device)
    duck_model.eval()
    print("DuckNet loaded successfully.")

    # Load Lama
    print("Loading Big-Lama...")
    predict_config = OmegaConf.load(args.lama_config)
    predict_config.model.path = args.lama_path
    train_config_path = Path(args.lama_path) / "config.yaml"

    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    checkpoint_path = Path(args.lama_path) / "models" / predict_config.model.checkpoint
    lama_model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location="cpu"
    )
    lama_model.freeze()
    lama_model.to(device)
    print("Big-Lama loaded successfully.")

    # Load StableDelight if not disabled
    pipe = None
    if not args.disable_sd:
        print("Loading StableDelight...")
        pipe = YosoDelightPipeline.from_pretrained(
            args.sd_path,
            trust_remote_code=True,
            variant="fp16",
            torch_dtype=torch.float16,
            t_start=0,
        ).to(device)
        print("StableDelight loaded successfully.")

    return device, duck_model, lama_model, pipe


# ---------------------- Prediction Loop ----------------------
@torch.no_grad()
def predict(args):
    device, duck_model, lama_model, pipe = build_models(args)

    input_paths = sorted(glob.glob(str(Path(args.input) / "*")))
    output_dataloader = dataloaders.get_dataloaders_output(input_paths)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    for idx, input_img in enumerate(output_dataloader):
        img_name = Path(input_paths[idx]).stem
        print(f"[{idx+1}/{len(output_dataloader)}] Processing: {img_name}")

        input_img = input_img.to(device)
        w, h = input_img.shape[2], input_img.shape[3]
        pad_h, pad_w = (32 - h % 32) % 32, (32 - w % 32) % 32

        # Step 1: DuckNet Prediction
        if pad_w or pad_h:
            resized_img = F.interpolate(
                input_img,
                size=(w + pad_w, h + pad_h),
                mode="bilinear",
                align_corners=False,
            )
            output = duck_model(resized_img)
            output = F.interpolate(
                output, size=(w, h), mode="bilinear", align_corners=False
            )
        else:
            output = duck_model(input_img)

        specular_mask = (np.squeeze(output.cpu().numpy()) > 0).astype(np.uint8) * 255

        if args.trad_help:
            img_np = np.clip(
                (input_img[0].permute(1, 2, 0).cpu().numpy()) * 255, 0, 255
            ).astype("uint8")
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            specular_mask[gray > 230] = 255

        # Step 2: Lama Inpainting
        if args.lama_dilate_size:
            mask = dilate_mask(specular_mask, args.lama_dilate_size)
        else:
            mask = specular_mask

        mask_tensor = torch.from_numpy(mask).float()[None, None]
        batch = {
            "image": (input_img + 1) / 2,
            "mask": mask_tensor,
        }
        unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
        batch["image"] = pad_tensor_to_modulo(batch["image"], 8)
        batch["mask"] = pad_tensor_to_modulo(batch["mask"], 8)
        batch = move_to_device(batch, device)
        batch["mask"] = (batch["mask"] > 0).float()

        batch = lama_model(batch)
        predict_config = OmegaConf.load(args.lama_config)
        inpainted_img = batch[predict_config.out_key][0].permute(1, 2, 0).cpu().numpy()
        inpainted_img = inpainted_img[: unpad_to_size[0], : unpad_to_size[1]]
        inpainted_img = np.clip(inpainted_img * 255, 0, 255).astype("uint8")

        # Step 3: StableDelight Post-process
        if args.disable_sd:
            output_img = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR)
        else:
            output_img = process_with_sd(inpainted_img, args, pipe)

        cv2.imwrite(str(Path(args.output) / f"{img_name}.png"), output_img)


# ---------------------- Argument Parsing ----------------------
def get_args():
    parser = argparse.ArgumentParser(description="Image Specular Removal Pipeline")
    parser.add_argument(
        "--input", type=str, default="./Test", help="Input image directory"
    )
    parser.add_argument(
        "--output", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--duck_path", type=str, required=True, help="Path to DuckNet checkpoint"
    )
    parser.add_argument(
        "--sd_path", type=str, required=True, help="Path to StableDelight checkpoint"
    )
    parser.add_argument(
        "--lama_path", type=str, required=True, help="Path to Big-Lama checkpoint"
    )
    parser.add_argument(
        "--lama_config", type=str, default="./configs/lama_predict_default.yaml"
    )
    parser.add_argument(
        "--lama_dilate_size",
        type=int,
        default=12,
        help="Dilation kernel size for Lama mask",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1536,
        help="Processing resolution for StableDelight",
    )
    parser.add_argument(
        "--mix_thresh",
        type=int,
        default=190,
        help="Threshold for overexposed region detection",
    )
    parser.add_argument("--mix_dilate_size", type=int, default=15)
    parser.add_argument("--mix_blur_size", type=int, default=55)
    parser.add_argument("--mix_blur_sigma", type=int, default=10)
    parser.add_argument(
        "--disable_sd", action="store_true", help="Disable StableDelight refinement"
    )
    parser.add_argument(
        "--trad_help", action="store_true", help="Enable traditional specular assist"
    )
    return parser.parse_args()


# ---------------------- Entry Point ----------------------
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()
