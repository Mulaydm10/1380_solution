import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import math
import random
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from safetensors.torch import load_file

from config import (
    model as model_config_dict,
    scheduler as scheduler_config_dict,
    vae as vae_config_dict,
)
from src.data.collate import Collate
from src.data.dataset import SensorGenDataset
from src.models.stdit.control_embedder import ControlEmbedder
from src.models.stdit.stdit3 import STDiT3, STDiT3Config
from src.registry import MODELS, SCHEDULERS, build_module
from src.schedulers.rf.rectified_flow import RFlowScheduler


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
            if 'weight' in key:
                nn.init.kaiming_uniform_(model_value, a=math.sqrt(5))
            else:
                nn.init.zeros_(model_value)

    model.load_state_dict(model_dict)
    print(f"Successfully loaded {loaded_count} matching keys.")
    print(f"Skipped and re-initialized {len(skipped_keys)} mismatched keys: {skipped_keys}")
    return loaded_count, skipped_keys

def save_gif(images, path, fps=2):
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
    parser.add_argument("--data_dir", type=str, default="/content/dataset/test_data_300/")
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
    vae_state_dict = load_file("/content/1380-solution_github/checkpoints/CogVideoX-2b/vae/diffusion_pytorch_model.safetensors", device=device)
    # Add the 'module.' prefix to match the model's architecture
    vae_state_dict_with_prefix = {f'module.{k}': v for k, v in vae_state_dict.items()}
    vae.load_state_dict(vae_state_dict_with_prefix)
    vae.eval()

    # Create a config object and then instantiate the model
    config = STDiT3Config(**model_config_dict)
    model = STDiT3(config).to(device, dtype)
    # Dynamically find the latest checkpoint from training_checkpoints
    training_checkpoints_dir = Path("/content/1380-solution_github/training_checkpoints")
    epoch_dirs = [d for d in training_checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]

    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories found in {training_checkpoints_dir}")

    # Sort by epoch number to get the latest
    latest_epoch_dir = sorted(epoch_dirs, key=lambda p: int(p.name.split('_')[1]))[-1]
    latest_checkpoint_path = latest_epoch_dir / "model.safetensors"

    print(f"Loading main model from latest checkpoint: {latest_checkpoint_path}")
    partial_load_checkpoint(model, latest_checkpoint_path, map_location=device)
    model.eval()

    scheduler = build_module(scheduler_config_dict, SCHEDULERS)

    embedder = ControlEmbedder(MODELS, **model.config.__dict__).to(device, dtype)
    embedder.eval()

    # 2. Get all scenes and randomly select a subset
    all_scene_paths = [p for p in Path(args.data_dir).iterdir() if p.is_dir()]
    random.shuffle(all_scene_paths)
    selected_scenes = all_scene_paths[: args.num_scenes]
    print(f"Found {len(all_scene_paths)} total scenes. Randomly selected {len(selected_scenes)} for inference.")

    # 3. Main Inference Loop
    for scene_path in tqdm(selected_scenes, desc="Processing Scenes"):
        try:
            print(f"\n--- Processing scene: {scene_path.name} ---")
            # Load data for the current scene
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

            # Move data to device
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(device, dtype)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            batch_data[key][sub_key] = [t.to(device, dtype) for t in sub_value]

            # Generate conditioning embedding
            with torch.no_grad():
                            if 'bev_grid' not in batch_data:
                                raise ValueError("BEV missing – Check dataset.py raster call")
                            cond_emb = embedder(
                                batch_data['bboxes_3d_data'],
                                batch_data['camera_param'],
                                bev_grid=batch_data['bev_grid'],
                            )
            # Denoising loop
            latents = torch.randn((1, 5, 80, 32, 32), device=device, dtype=dtype)
            scheduler.set_timesteps(args.steps, device=device)

            for i, t in enumerate(scheduler.timesteps):
                with torch.no_grad():
                    t_batch = t.repeat(latents.shape[0]).to(device)
                    noise_pred = model(latents, t_batch, encoder_hidden_states=cond_emb)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Decode and save
            with torch.no_grad():
                latents = latents / vae.config.scaling_factor
                decoded_frames = []
                for i in range(latents.size(1)):
                    single_view_latent = latents[:, i, :, :, :].unsqueeze(2)
                    decoded_view = vae.decode(single_view_latent).sample
                    decoded_frames.append(decoded_view)
                images = torch.cat(decoded_frames, dim=1)

            output_gif_path = os.path.join(args.output_dir, f"{scene_path.name}.gif")
            save_gif(images, output_gif_path)

        except Exception as e:
            print(f"!!! Failed to process scene {scene_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("--- All scenes processed. ---")
