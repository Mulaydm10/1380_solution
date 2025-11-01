import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from src.data.dataset import SensorGenDataset
from src.data.collate import Collate
from src.registry import MODELS, SCHEDULERS, build_module
from config import model as model_config_dict, vae as vae_config_dict, scheduler as scheduler_config_dict

def main():
    # --- Configuration ---
    data_path = '/content/dataset/test_data_300/'
    num_test_scenes = 20
    batch_size = 1

    # --- Setup ---
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device

    # --- Dataloader ---
    scenes = [p.name for p in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, p))][:num_test_scenes]
    dataset = SensorGenDataset(scenes, data_path, mode='train')
    collate_fn = Collate()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2)

    # --- Models ---
    vae = build_module(vae_config_dict, MODELS)
    model = build_module(model_config_dict, MODELS)
    scheduler = build_module(scheduler_config_dict, SCHEDULERS)
    
    # For this test, we don't need to load pre-trained weights or have an optimizer
    # We just want to test the forward pass with real data

    # --- Prepare with Accelerate ---
    model, vae, dataloader = accelerator.prepare(model, vae, dataloader)

    model.train()
    vae.eval() # VAE is not being trained

    losses = []
    print(f"--- Running BEV Integration Test on {num_test_scenes} scenes ---")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            # Move VAE to GPU for encoding, then back to CPU
            vae.to(device)
            # Assuming first image in gt stack is the target
            gt_video_for_latent = batch['images_gt'][:, 0:1, :, :, :].permute(0, 2, 1, 3, 4)
            gt_latents = vae.encode(gt_video_for_latent)
            
            # Encode conditioning cameras
            B, N_COND, C, H, W = batch['cond_cam_raw'].shape
            cond_reshaped = batch['cond_cam_raw'].view(B * N_COND, C, 1, H, W)
            cond_latents = vae.encode(cond_reshaped)
            # Reshape back and combine channels
            cond_latents = cond_latents.view(B, N_COND, cond_latents.shape[1], cond_latents.shape[2], cond_latents.shape[3], cond_latents.shape[4])
            cond_cam_latents = cond_latents.view(B, N_COND * cond_latents.shape[2], cond_latents.shape[3], cond_latents.shape[4], cond_latents.shape[5])
            vae.to("cpu")

        # Create noise and timesteps
        noise = torch.randn_like(gt_latents)
        timesteps = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
        noisy_latents = scheduler.add_noise(gt_latents, noise, timesteps)

        # Prepare conditioning for the model
        # This part is simplified as we don't have a full control_embedder yet
        # We pass the BEV grid directly to the model
        bbox_data = batch['bboxes_3d_data']
        camera_params = batch['camera_param']
        bev_grid = batch['bev_grid']

        # Forward pass
        with accelerator.autocast():
            predicted_noise = model(
                noisy_latents,
                timesteps,
                cond_cam=cond_cam_latents,
                bbox=bbox_data,
                cams=camera_params, # This might need adjustment based on how control_embedder is built
                height=torch.tensor([H], device=device),
                width=torch.tensor([W], device=device),
                NC=5, # As per Grok's analysis
                bev_grid=bev_grid # Pass the BEV grid
            )
            loss = F.mse_loss(predicted_noise.float(), noise.float())
        
        losses.append(loss.item())
        print(f"Step {i+1}/{num_test_scenes}, Loss: {loss.item():.4f}")

    avg_loss = np.mean(losses)
    print(f"\n--- BEV Test Complete ---")
    print(f"Average Loss over {num_test_scenes} scenes: {avg_loss:.4f}")
    # Grok's target: Compare this to a no-BEV baseline run. A lower loss is a good sign.

if __name__ == "__main__":
    main()
