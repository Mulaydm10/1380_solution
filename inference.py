import os
import pathlib
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from mmengine.config import Config
from src.registry import MODELS, SCHEDULERS, build_module
from src.data.dataset import SensorGenDataset
from src.data.collate import Collate
from src.data.types import sanitize_camera_name
from src.utils.inference_utils import prepare_control_data, inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()
    return args


def unprocess_pixel_values(x: torch.Tensor) -> torch.Tensor:
    # x is in [-1, 1]
    x = (x + 1) / 2
    x = (x * 255).clamp(0, 255).to(torch.uint8)
    return x


def sanitize_filename(name: str) -> str:
    return name.replace("-", "").replace(":", "")


def save_numpy_image_as_jpg(img_array, filename):
    """
    Сохраняет numpy-массив img_array как jpg-файл с именем filename.

    img_array: numpy.ndarray, shape (H, W) или (H, W, 3)
    filename: строка, имя файла (например, 'output.jpg')
    """
    if img_array.dtype != np.uint8:
        # при необходимости преобразуйте расширенный диапазон в [0,255]
        img_array = 255 * (img_array - img_array.min()) / (img_array.ptp() + 1e-8)
        img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    img = img.resize((512, 288), Image.LANCZOS)
    img.save(filename, format="JPEG")


def main():
    args = parse_args()
    data_path = pathlib.Path(args.data_path)
    if args.save_dir == None:
        args.save_dir = data_path
    scenes = [p.name for p in data_path.iterdir() if (p / "ride_id.json").exists()][:20]
    print(f"Found {len(scenes)} scenes")

    cfg = Config.fromfile(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # dataset
    dataset = SensorGenDataset(scenes, data_path)
    collate_fn = Collate()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # models
    vae = build_module(cfg.get("vae", None), MODELS)
    if vae is not None:
        vae = vae.to(device, dtype=dtype).eval()
    latent_size = (None, None, None)
    vae_out_channels = cfg.get("vae_out_channels", 16)

    # == build diffusion model ==
    model = (
        build_module(
            cfg.get("model"),
            MODELS,
            input_size=latent_size,
            in_channels=vae_out_channels,
        )
        .to(device, dtype=dtype)
        .eval()
    )

    state_dict = torch.load(args.ckpt_path, weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    # sheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    seed = 42
    generator = torch.Generator("cpu").manual_seed(seed)

    for batch in tqdm(dataloader):
        B, T, NC = batch["camera_param"].shape[:3]

        with torch.no_grad():
            model_args = prepare_control_data(batch, vae, device, dtype, B, T, NC)
            samples = inference(
                model,
                scheduler,
                vae,
                model_args,
                device,
                dtype,
                generator
            )
            
        # save images
        for i in range(B):
            save_dir = f"ride_{sanitize_filename(batch['ride_id'][i].ride_date)}_{sanitize_filename(batch['ride_id'][i].ride_time)}_{sanitize_filename(batch['ride_id'][i].log_time)}_{sanitize_filename(batch['ride_id'][i].rover)}_{str(batch['ride_id'][i].message_ts)}"
            save_path = os.path.join(args.save_dir, save_dir, "images_pred")
            os.makedirs(save_path, exist_ok=True)
            for j, camera_name in enumerate(batch["camera_names"][i]):
                save_numpy_image_as_jpg(
                    unprocess_pixel_values(samples[i, j, :, 0])
                    .data.cpu()
                    .numpy()
                    .transpose(1, 2, 0),
                    os.path.join(save_path, f"camera_{sanitize_camera_name(camera_name)}.jpg"),
                )


if __name__ == "__main__":
    main()