import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import math
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from config import (
    model as model_config_dict,
    scheduler as scheduler_config_dict,
    vae as vae_config_dict,
)
from src.data.collate import Collate
from src.data.dataset import SensorGenDataset
from src.models.stdit.control_embedder import ControlEmbedder
from src.models.stdit.stdit3 import STDiT3
from src.registry import MODELS, build_module
from src.schedulers.rf.rectified_flow import RectifiedFlowScheduler


def partial_load_checkpoint(model, checkpoint_path):
    print("=== Partial Checkpoint Load START ===")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    loaded_count = 0
    skipped_keys = []

    for key, ckpt_value in state_dict.items():
        if key not in model_dict:
            print(f"[INFO] Unexpected key in checkpoint (skipping): {key}")
            continue

        model_value = model_dict[key]

        if ckpt_value.shape == model_value.shape:
            model_dict[key].copy_(ckpt_value)
            loaded_count += 1
        else:
            skipped_keys.append(key)
            if 'weight' in key:
                nn.init.kaiming_uniform_(model_value, a=math.sqrt(5))
            else:
                nn.init.zeros_(model_value)

    model.load_state_dict(model_dict)
    print(f"Successfully loaded {loaded_count} matching keys.")
    print(f"Skipped and re-initialized {len(skipped_keys)} mismatched keys: {skipped_keys}")
    return loaded_count, skipped_keys

def save_gif(images, path, camera_names=None, fps=2):
    images = (images.clamp(-1, 1) + 1) / 2
    images = images.squeeze(0)
    pil_frames = [
        Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        for img in images
    ]

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"GIF saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, default="/content/dataset/test_data_300/scene-0")
    parser.add_argument("--output_path", type=str, default="gen_scene.gif")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Initialize Components
    print("--- Initializing VAE, Model, Scheduler, and Embedder ---")
    vae = build_module(vae_config_dict, MODELS).to(device, dtype)
    vae.load_state_dict(torch.load("/content/1380-solution_github/checkpoints/CogVideoX-2b/vae/diffusion_pytorch_model.safetensors", map_location=device))
    vae.eval()

    model = STDiT3(**model_config_dict).to(device, dtype)
    partial_load_checkpoint(model, "/content/1380-solution_github/checkpoints/ckpt/ema.pt")
    model.eval()

    scheduler = RectifiedFlowScheduler(**scheduler_config_dict)

    embedder = ControlEmbedder(MODELS, **model.config.__dict__).to(device, dtype)
    embedder.eval()

    # 2. Load Scene Data using SensorGenDataset and Collate
    print(f"--- Loading data for scene: {args.scene_path} ---")
    scene_path = Path(args.scene_path)
    ds = SensorGenDataset(
        scenes=[scene_path.name],
        data_root=scene_path.parent,
        mode='inference',
        num_cond_cams=4,
        image_size=(512, 512),
    )
    scene_data = ds[0]
    collate_fn = Collate()
    batch_data = collate_fn([scene_data])

    # Move all data to the correct device
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            batch_data[key] = value.to(device, dtype)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                     # Handle list of tensors for bboxes_3d_data
                    batch_data[key][sub_key] = [t.to(device, dtype) for t in sub_value]

    # 3. Generate Conditioning Embedding
    print("--- Generating conditioning embedding ---")
    with torch.no_grad():
        cond_emb = embedder(
            batch_data['bboxes_3d_data'],
            batch_data['camera_param'],
            bev_grid=batch_data['bev_grid'],
        )
    print(f"Conditioning embedding shape: {cond_emb.shape}")

    # 4. Inference Pipeline
    print(f"--- Starting inference loop for {args.steps} steps ---")
    latents = torch.randn((1, 5, 80, 32, 32), device=device, dtype=dtype)
    scheduler.set_timesteps(args.steps, device=device)

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        with torch.no_grad():
            t_batch = t.repeat(latents.shape[0]).to(device)
            noise_pred = model(latents, t_batch, encoder_hidden_states=cond_emb)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 5. Decode and Save
    print("--- Decoding latents and saving GIF ---")
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        decoded_frames = []
        for i in range(latents.size(1)):
            single_view_latent = latents[:, i, :, :, :].unsqueeze(2)
            decoded_view = vae.decode(single_view_latent).sample
            decoded_frames.append(decoded_view)
        images = torch.cat(decoded_frames, dim=1)

    camera_names = ['front_middle', 'rear', 'left_backward', 'left_forward', 'right_forward']
    save_gif(images, args.output_path, camera_names=camera_names)

    print(f"--- Inference Complete: {args.output_path} saved. ---")