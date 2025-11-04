import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from config import model as model_config_dict
from config import scheduler as scheduler_config_dict
from config import vae as vae_config_dict
from src.data.collate import Collate
from src.data.dataset import SensorGenDataset
from src.models.stdit.control_embedder import ControlEmbedder
from src.models.stdit.stdit3 import STDiT3, STDiT3Config
from src.registry import MODELS, SCHEDULERS, build_module
from src.schedulers.rf.rectified_flow import RFlowScheduler

# Replace RFlowScheduler with DDPMScheduler
scheduler_config_dict["type"] = "diffusers.DDPMScheduler"
scheduler_config_dict["beta_schedule"] = "squaredcos_cap_v2"

# Remove RFlowScheduler-specific arguments
if "use_timestep_transform" in scheduler_config_dict:
    del scheduler_config_dict["use_timestep_transform"]
if "transform_scale" in scheduler_config_dict:
    del scheduler_config_dict["transform_scale"]


class RFlowScheduler:
    pass


def partial_load_checkpoint(model, checkpoint_path, map_location):
    print("=== Partial Checkpoint Load START ===")
    state_dict = load_file(checkpoint_path, device=map_location)
    model_dict = model.state_dict()
    loaded_count = 0
    skipped_keys = []

    for key, ckpt_value in state_dict.items():
        if key not in model_dict:
            continue

        model_value = model_dict[key]

        if ckpt_value.shape == model_value.shape:
            model_dict[key].copy_(ckpt_value)
            loaded_count += 1
        else:
            skipped_keys.append(key)
            if "weight" in key:
                nn.init.kaiming_uniform_(model_value, a=math.sqrt(5))
            else:
                nn.init.zeros_(model_value)

    model.load_state_dict(model_dict)
    print(f"Successfully loaded {loaded_count} matching keys.")
    print(
        f"Skipped and re-initialized {len(skipped_keys)} mismatched keys: {skipped_keys}"
    )
    return loaded_count, skipped_keys


def save_gif(images_tensor, path='gen.gif', camera_names=None, fps=1):
    """
    From [1,3,320,256,256] → 320 frames (256,256,3) numpy → GIF.
    """
    # Denorm to [0,1]
    images_tensor = torch.clamp(images_tensor, -1, 1)
    images_tensor = (images_tensor + 1) / 2  # [1,3,320,256,256]

    # Squeeze batch, permute to [320,256,256,3]
    images = images_tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # V=320 first, then H,W,C

    # V-loop to frames
    frames = []
    for v in range(images.shape[0]):  # 320 views/frames
        img_v = images[v]  # (256,256,3) float [0,1]
        img_v = (img_v * 255).astype(np.uint8)  # uint8 for PIL
        frames.append(Image.fromarray(img_v))

    # Save GIF (loop=0 for repeat; duration=1000/fps ms)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"GIF saved: {path} ({len(frames)} frames at {fps}fps)")

    # Optional: PNG seq for eval (named views if camera_names)
    if camera_names and len(camera_names) == len(frames):
        for i, (v, name) in enumerate(zip(frames, camera_names)):
            v.save(f"{name}.png")
        print(f"PNG seq saved: view_{camera_names[0]}.png etc.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/content/dataset/test_data_300/"
    )
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--num_scenes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize Components
    print("--- Initializing VAE, Model, Scheduler, and Embedder ---")
    vae = build_module(vae_config_dict, MODELS).to(device, dtype)
    vae_state_dict = load_file(
        "/content/1380-solution_github/checkpoints/CogVideoX-2b/vae/diffusion_pytorch_model.safetensors",
        device=device,
    )
    # Add the 'module.' prefix to match the model's architecture
    vae_state_dict_with_prefix = {f"module.{k}": v for k, v in vae_state_dict.items()}
    vae.load_state_dict(vae_state_dict_with_prefix)
    vae.eval()

    # Offload VAE to CPU
    vae = vae.cpu()

    # Create a config object and then instantiate the model
    model_config_dict["input_size"] = (16, 80, 32, 32)
    model_config_dict["patch_size"] = (1, 2, 2)
    model_config_dict["enable_flash_attn"] = True
    config = STDiT3Config(**model_config_dict)
    model = STDiT3(config).to(device, dtype)
    # Dynamically find the latest checkpoint from training_checkpoints
    training_checkpoints_dir = Path(
        "/content/1380-solution_github/training_checkpoints"
    )
    epoch_dirs = [
        d
        for d in training_checkpoints_dir.iterdir()
        if d.is_dir() and d.name.startswith("epoch_")
    ]

    if not epoch_dirs:
        raise FileNotFoundError(
            f"No epoch directories found in {training_checkpoints_dir}"
        )

    # Sort by epoch number to get the latest
    latest_epoch_dir = sorted(epoch_dirs, key=lambda p: int(p.name.split("_")[1]))[-1]
    latest_checkpoint_path = latest_epoch_dir / "model.safetensors"

    partial_load_checkpoint(model, latest_checkpoint_path, map_location=device)
    model.eval()


# Scheduler: Clean DDPM config (RF bleed ignored)
ddpm_config = {
    'num_train_timesteps': 1000,
    'beta_schedule': 'squaredcos_cap_v2',  # Smooth betas for stable diffusion
    'prediction_type': 'epsilon',  # Noise prediction (match model)
    'trained_betas': None,  # Use default
    'variance_type': 'fixed_small',  # Stable variance
    'clip_sample': True,  # Clip noise for stability
    'thresholding': False,  # No threshold
}
scheduler = DDPMScheduler(**ddpm_config)

# Set steps for loop (tune 8-50)
scheduler.set_timesteps(args.steps, device=device)  # Low for fast test; high for quality

# Initialize components inside main block
if __name__ == "__main__":
    # 2. Get all scenes and randomly select a subset
    all_scene_paths = [p for p in Path(args.data_dir).iterdir() if p.is_dir()]
    random.shuffle(all_scene_paths)
    selected_scenes = all_scene_paths[: args.num_scenes]
    print(
        f"Found {len(all_scene_paths)} total scenes. Randomly selected {len(selected_scenes)} for inference."
    )

    # 3. Initialize embedder
    embedder = ControlEmbedder(MODELS, **model.config.__dict__).to(device, dtype)
    embedder.eval()

    # Offload embedder to CPU
    embedder = embedder.cpu()

    # 4. Main Inference Loop
    for scene_path in tqdm(selected_scenes, desc="Processing Scenes"):
        try:
            print(f"\n--- Processing scene: {scene_path.name} ---")
            # Load data for the current scene
            ds = SensorGenDataset(
                scenes=[scene_path.name],
                data_root=scene_path.parent,
                mode="inference",
                num_cond_cams=4,
                image_size=(512, 512),
            )
            scene_data = ds[0]

            # Manually create batch to mimic default collate used in train.py
            batch_data = {
                "bboxes_3d_data": {
                    "bboxes": [scene_data["bboxes_3d_data"]["bboxes"]],
                    "classes": [scene_data["bboxes_3d_data"]["classes"]],
                    "masks": [scene_data["bboxes_3d_data"]["masks"]],
                },
                "camera_param": torch.as_tensor(scene_data["camera_param"]).unsqueeze(
                    0
                ),
                "bev_grid": torch.as_tensor(scene_data["bev_grid"]).unsqueeze(0),
                "ride_id": [scene_data["ride_id"]],
            }

            # Move data to device
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(device, dtype)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            batch_data[key][sub_key] = [
                                torch.as_tensor(t).to(device, dtype) for t in sub_value
                            ]

            # Generate conditioning embedding
            with torch.no_grad():
                embedder.to(device)
                cond_emb = embedder(
                    batch_data["bboxes_3d_data"],
                    batch_data["camera_param"],
                    bev_grid=batch_data["bev_grid"],
                )
                embedder.cpu()
                print(
                )

            # Denoising loop
            latents = torch.randn((1, 16, 80, 32, 32), device=device, dtype=dtype)
            for i, t in enumerate(scheduler.timesteps):
                with torch.no_grad():
                    t_batch = t.repeat(latents.shape[0]).to(device)
                    noise_pred = model(latents, t_batch, encoder_hidden_states=cond_emb)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                print(f"Step {i+1}/{args.steps} – Timestep: {t.item():.2f}")

            # --- Start Grok Patch: VAE Decode & Save ---
            # Fix 1: Reshape Latent Tensor if fused
            # Assuming the model output `latents` might have a fused shape.
            # The target is [B, V, C, T_lat, H_lat, W_lat] but VAE expects [B*V, C, T_lat, H_lat, W_lat]
            # Based on STDiT3, the output shape is [1, 16, 80, 32, 32]. Let's assume this represents 5 views.
            # The temporal dimension is 80, and there are 5 views. Let's assume T=16 frames per view.
            # New shape should be (B, V, C, T_frames, H, W). Let's try to decode view by view.
            
            print(f"Initial latents shape: {latents.shape}")
            latents = latents / vae.scaling_factor
            print(f"Scaled latents min/max: {latents.min():.3f}/{latents.max():.3f}")

            # Reshape for multi-view decoding
            # The model output is likely (B, C, T, H, W). Let's assume T contains the views.
            # For this model, let's assume 5 views are packed in the temporal dimension.
            num_views = 5
            if latents.shape[2] % num_views == 0:
                b, c, t, h, w = latents.shape
                frames_per_view = t // num_views
                latents_reshaped = latents.view(b, c, num_views, frames_per_view, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()
                latents_for_decode = latents_reshaped.view(b * num_views, c, frames_per_view, h, w)
                print(f"Reshaped latents for VAE: {latents_for_decode.shape}")
            else:
                latents_for_decode = latents # Fallback

            with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
                decoded = vae.decode(latents_for_decode)
            print(f"Decoded shape/range: {decoded.shape}, min {decoded.min():.3f}, max {decoded.max():.3f}")

            # Fix 3: Soft Norm + Clamp
            decoded_soft = torch.clamp(decoded, -1.5, 1.5)
            decoded_norm = (decoded_soft / 2 + 0.5).clamp(0, 1)
            print(f"Soft norm range: min {decoded_norm.min():.3f}, max {decoded_norm.max():.3f}")

            # To uint8 and save PNG sequence
            decoded_uint8 = (decoded_norm * 255).round().byte().cpu()
            print(f"Uint8 shape: {decoded_uint8.shape}")

            # Save PNG Seq for each view
            for i in range(decoded_uint8.shape[0]): # Iterate through views
                view_frames = decoded_uint8[i] # Shape (C, T, H, W)
                for j in range(view_frames.shape[1]): # Iterate through frames in the view
                    frame_tensor = view_frames[:, j, :, :].permute(1, 2, 0) # H, W, C
                    img = Image.fromarray(frame_tensor.numpy())
                    os.makedirs(os.path.join(args.output_dir, scene_path.name), exist_ok=True)
                    img.save(os.path.join(args.output_dir, scene_path.name, f"view_{i}_frame_{j}.png"))
            print(f"Saved PNG sequence for scene {scene_path.name}")
            # --- End Grok Patch ---

        except Exception as e:
            print(f"!!! Failed to process scene {scene_path.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("--- All scenes processed. ---")
