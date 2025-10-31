
import os
import sys
# Add project root to sys.path to enable imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gc
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import pathlib

# Project imports (assuming they are in PYTHONPATH)
from src.data.dataset import SensorGenDataset # Using existing dataset
from src.data.collate import Collate
from src.models.stdit.stdit3 import STDiT3
from src.models.cog_vae.vae_cogvideox import VideoAutoencoderKLCogVideoX
from src.registry import MODELS, SCHEDULERS, build_module
from config import model as model_config_dict, vae as vae_config_dict, scheduler as scheduler_config_dict # Import configs

# --- Training Configuration (can be moved to config.py later) ---
class TrainingConfig:
    num_epochs = 1
    batch_size = 1 # Keep at 1 for now due to VRAM constraints
    gradient_accumulation_steps = 4 # Simulate batch size of 4
    learning_rate = 1e-5
    lr_warmup_steps = 500
    mixed_precision = 'fp16' # Use fp16 for T4 compatibility
    output_dir = './training_checkpoints'
    seed = 42
    max_grad_norm = 1.0
    data_path = '/content/dataset/test_data_300/' # Colab path to dataset
    num_workers = 2 # For DataLoader

training_config = TrainingConfig()

def main():
    # 1. SETUP
    set_seed(training_config.seed)
    accelerator = Accelerator(
        mixed_precision=training_config.mixed_precision,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(training_config.output_dir, "logs")
    )
    
    if accelerator.is_main_process:
        os.makedirs(training_config.output_dir, exist_ok=True)

    # 2. DATALOADERS
    # We need to modify SensorGenDataset to return ground truth images for training
    # For now, it will just load what it loads, and we'll adapt it in Day 2/3
    scenes = [p.name for p in pathlib.Path(training_config.data_path).iterdir() if (p / "ride_id.json").exists()]
    train_dataset = SensorGenDataset(scenes, pathlib.Path(training_config.data_path)) 
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
    )

    # 3. MODELS
    # Load VAE
    vae = build_module(vae_config_dict, MODELS)
    # Load STDiT3 model
    model = build_module(model_config_dict, MODELS)

    # Load pre-trained weights for STDiT3
    # Assuming the checkpoint is in the same structure as inference.py expects
    # This path needs to be relative to the Colab environment where the checkpoint is copied
    ckpt_path = "/content/1380-solution_github/checkpoints/ckpt/ema.pt" # Path to pre-trained weights in local Colab storage
    state_dict = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    # Freeze VAE and move to CPU to save VRAM
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    vae.to("cpu") # Keep VAE on CPU, move to GPU only when needed

    # 4. OPTIMIZER & SCHEDULER
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=0.0)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * training_config.num_epochs),
    )

    # 5. ACCELERATE PREPARE
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 6. TRAINING LOOP
    if accelerator.is_main_process:
        print("***** Starting training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num epochs = {training_config.num_epochs}")
        print(f"  Batch size (effective) = {training_config.batch_size * training_config.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {len(train_dataloader) * training_config.num_epochs}")

    global_step = 0
    for epoch in range(training_config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # OOM Guard: Clear cache before forward pass
                torch.cuda.empty_cache()
                gc.collect()

                # === DUMMY DATA (Replace with real batch later) ===
                B, T, C, H, W = 1, 1, 3, 256, 256
                NC = 5  # Total cameras: 1 front (GT) + 4 side
                N_COND = NC - 1  # = 4 → conditioning cameras
                device = accelerator.device

                # === 1. Dummy GT video (front camera) ===
                dummy_gt_video = torch.randn(B, C, T, H, W, device=device)

                # === 2. Dummy conditioning videos (4 side cameras) ===
                dummy_cond_cam_raw = torch.randn(B, N_COND, C, T, H, W, device=device)  # (1, 4, 3, 1, 256, 256)

            with accelerator.accumulate(model):
                # OOM Guard: Clear cache before forward pass
                torch.cuda.empty_cache()
                gc.collect()

                # === DUMMY DATA (Replace with real batch later) ===
                B, T, C, H, W = 1, 1, 3, 256, 256
                NC = 5  # Total cameras: 1 front (GT) + 4 side
                N_COND = NC - 1  # = 4 → conditioning cameras
                device = accelerator.device

                # === 1. Dummy GT video (front camera) ===
                dummy_gt_video = torch.randn(B, C, T, H, W, device=device)

                # === 2. Dummy conditioning videos (4 side cameras) ===
                dummy_cond_cam_raw = torch.randn(B, N_COND, C, T, H, W, device=device)  # (1, 4, 3, 1, 256, 256)

                # === 3. Move VAE to GPU and encode ===
                vae.to(device)
                with torch.no_grad():
                    # Encode GT: (B, C, T, H, W) → (B, 4, T', H', W') e.g., (1, 4, 1, 64, 64)
                    latents_gt = vae.encode(dummy_gt_video)
                    B_gt, C_gt, latent_T, latent_H, latent_W = latents_gt.shape  # Unpack all dims

                    # Encode conditioning: (B*N_COND, C, T, H, W) → (B*N_COND, 4, T', H', W') e.g., (4, 4, 1, 64, 64)
                    cond_reshaped = dummy_cond_cam_raw.view(B * N_COND, C, T, H, W)
                    cond_latents = vae.encode(cond_reshaped)

                    # Verify dims match GT
                    B_cond, C_cond, T_cond, H_cond, W_cond = cond_latents.shape
                    assert C_cond == C_gt and T_cond == latent_T and H_cond == latent_H and W_cond == latent_W, f"Latent dims mismatch: GT {latents_gt.shape} vs Cond {cond_latents.shape}"

                    # Reshape cond: (B*N_COND, 4, T', H', W') → (B, N_COND, 4, T', H', W') e.g., (1, 4, 4, 1, 64, 64)
                    cond_latents = cond_latents.view(B, N_COND, C_cond, latent_T, latent_H, latent_W)

                    # Collapse channels: (B, N_COND*4, T', H', W') e.g., (1, 16, 1, 64, 64)
                    cond_cam_latents = cond_latents.view(B, N_COND * C_cond, latent_T, latent_H, latent_W)

                vae.to("cpu")
                torch.cuda.empty_cache()

                # === 4. Add noise ===
                noise = torch.randn_like(latents_gt)
                timesteps = torch.randint(0, 1000, (B_gt,), device=device).long()
                noisy_latents = latents_gt + noise

                # === 5. Dummy conditioning ===
                dummy_bbox = {
                    "bboxes": torch.randn(B, 1, 10, 8, 3, device=device),
                    "classes": torch.randint(0, 8, (B, 1, 10), device=device),
                    "masks": torch.ones(B, 1, 10, device=device)
                }
                dummy_cams = torch.randn(B, 1, 7, 3, 7, device=device)
                dummy_height = torch.tensor([H], device=device)
                dummy_width = torch.tensor([W], device=device)

                # === 6. Forward ===
                print(f"noisy_latents.shape: {noisy_latents.shape}")  # Debug: (1, 4, 1, 64, 64)
                print(f"cond_cam_latents.shape: {cond_cam_latents.shape}")  # Debug: (1, 16, 1, 64, 64)
                predicted_noise = model(
                    noisy_latents,  # (1, 4, 1, 64, 64)
                    timesteps,
                    cond_cam=cond_cam_latents,  # (1, 16, 1, 64, 64)
                    bbox=dummy_bbox,
                    cams=dummy_cams,
                    height=dummy_height,
                    width=dummy_width,
                    NC=NC  # 5
                )

                # Calculate loss (MSE for now)
                loss = F.mse_loss(predicted_noise, noise, reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging and saving
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                accelerator.log(logs, step=global_step)


        # End of epoch saving
        if accelerator.is_main_process:
            accelerator.save_state(os.path.join(training_config.output_dir, f"epoch_{epoch}"))


if __name__ == "__main__":
    main()
