import os
import sys

# Add project root to sys.path to enable imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import gc
import pathlib

import torch
import torch.nn.functional as F
from accelerate import Accelerator, cpu_offload
from accelerate.utils import set_seed
from einops import repeat
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

torch.backends.cudnn.benchmark = False
torch.backends.cuda.enable_flash_sdp(False)  # Standard attn, less VRAM





from config import model as model_config_dict  # Import configs
from config import scheduler as scheduler_config_dict
from config import vae as vae_config_dict
from src.data.collate import Collate

# Project imports (assuming they are in PYTHONPATH)
from src.data.dataset import SensorGenDataset  # Using existing dataset
from src.models.cog_vae.vae_cogvideox import VideoAutoencoderKLCogVideoX
from src.models.stdit.stdit3 import STDiT3
from src.registry import MODELS, SCHEDULERS, build_module


# --- Training Configuration (can be moved to config.py later) ---
class TrainingConfig:
    num_epochs = 1
    batch_size = 1  # Keep at 1 for now due to VRAM constraints
    gradient_accumulation_steps = 1  # Reduced to 1 to minimize memory
    learning_rate = 1e-5
    lr_warmup_steps = 500
    mixed_precision = "bf16"  # Stable for softmax, no fp32 cast
    output_dir = "./training_checkpoints"
    seed = 42
    max_grad_norm = 1.0
    data_path = "/content/dataset/test_data_300/"  # Colab path to dataset
    num_workers = 2  # For DataLoader


training_config = TrainingConfig()


def main():
    # 1. SETUP
    set_seed(training_config.seed)
    accelerator = Accelerator(
        mixed_precision=training_config.mixed_precision,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(training_config.output_dir, "logs"),
    )
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    if accelerator.is_main_process:
        os.makedirs(training_config.output_dir, exist_ok=True)

    # 2. DATALOADERS
    # We need to modify SensorGenDataset to return ground truth images for training
    # For now, it will just load what it loads, and we'll adapt it in Day 2/3
    scenes = [
        p.name
        for p in pathlib.Path(training_config.data_path).iterdir()
        if (p / "ride_id.json").exists()
    ]
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
    ckpt_path = "/content/1380-solution_github/checkpoints/ckpt/ema.pt"  # Path to pre-trained weights in local Colab storage
    state_dict = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    # --- Grok's Safeguard: Verify Fine-Tuning ---
    if accelerator.is_main_process:
        missing_keys, unexpected_keys = [], []
        for k in state_dict:
            if k not in model.state_dict():
                unexpected_keys.append(k)
        for k in model.state_dict():
            if k not in state_dict:
                missing_keys.append(k)
        
        print("--- Fine-Tuning Load Report ---")
        print(f"Loaded Matches: {len(model.state_dict()) - len(missing_keys)} / {len(model.state_dict())}")
        print(f"Missing Keys (New Layers): {len(missing_keys)}")
        print(f"Unexpected Keys (Old Layers): {len(unexpected_keys)}")
        
        spatial_loaded = all('spatial_blocks' in k for k in model.state_dict() if 'spatial_blocks' in k and k not in missing_keys)
        print(f"Core Spatial Blocks Preserved: {spatial_loaded}")
        if not spatial_loaded:
            raise RuntimeError("Critical Error: Core spatial blocks are not being loaded from checkpoint. Aborting.")
        print("---------------------------------")



    # Gradient checkpointing not supported for STDiT3, skip to avoid error
    # model.gradient_checkpointing_enable()

    # Freeze VAE and move to CPU to save VRAM
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    vae.to("cpu")  # Keep VAE on CPU, move to GPU only when needed

    # 4. OPTIMIZER & SCHEDULER
    optimizer = AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=0.0
    )

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
        print(
            f"  Batch size (effective) = {training_config.batch_size * training_config.gradient_accumulation_steps}"
        )
        print(
            f"  Total optimization steps = {len(train_dataloader) * training_config.num_epochs}"
        )

    global_step = 0
    for epoch in range(training_config.num_epochs):
        model.train()
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Ultra OOM Guard
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()


                with torch.no_grad():
                    vae.to(accelerator.device)
                    # The VAE can only encode one view at a time. We must loop through the 5 views.
                    # Original shape: [B, 5, C, H, W]
                    images_gt = batch['images_gt']
                    if accelerator.is_main_process:
                        print(f"[VAE Encode] Input images_gt shape: {images_gt.shape}")

                    latents_list = []
                    for i in range(images_gt.size(1)): # Loop through the 5 views
                        view = images_gt[:, i:i+1, :, :, :] # Get one view: [B, 1, C, H, W]
                        view_permed = view.permute(0, 2, 1, 3, 4) # Permute to [B, C, 1, H, W]
                        latent_view = vae.encode(view_permed)
                        latents_list.append(latent_view)
                    
                    gt_latents = torch.cat(latents_list, dim=1) # Stack along the view dimension: [B, 5, C_latent, H_latent, W_latent]
                    if accelerator.is_main_process:
                        print(f"[VAE Encode] Output gt_latents shape: {gt_latents.shape}")

                    vae.to("cpu")

                # RFLOW noise schedule - applied to all 5 views
                t = torch.rand(gt_latents.shape[0], device=accelerator.device).view(-1, 1, 1, 1, 1)
                noise = torch.randn_like(gt_latents) # Noise is now [B, 5, C, H, W]
                noisy_latents = (1 - t) * gt_latents + t * noise
                timesteps = t.squeeze(-1).squeeze(-1).squeeze(-1)

                # Prepare conditioning for the model
                from src.models.stdit.control_embedder import ControlEmbedder
                control_embedder_instance = ControlEmbedder(model.config)
                control_embedder_instance.to(accelerator.device)

                # --- Extensive Logging to Debug bboxes_3d_data ---
                if accelerator.is_main_process:
                    print("--- Extensive BBox Debug Log START ---")
                    def inspect_dict(d, indent=0):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                print(' ' * indent + f"Key: '{k}', Type: {type(v)}")
                                inspect_dict(v, indent + 2)
                            elif hasattr(v, 'shape'):
                                print(' ' * indent + f"Key: '{k}', Type: {type(v)}, Shape: {v.shape}, Dtype: {v.dtype}")
                            else:
                                print(' ' * indent + f"Key: '{k}', Type: {type(v)}, Value: {v}")
                    inspect_dict(batch['bboxes_3d_data'])
                    print("--- Extensive BBox Debug Log END ---")

                cond_emb = control_embedder_instance(
                    bboxes_dict=batch['bboxes_3d_data'],
                    camera_params=batch['camera_param'],
                    bev_grid=batch['bev_grid']
                )

                # Forward pass
                predicted_noise = model(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=cond_emb
                )
                loss = F.mse_loss(predicted_noise.float(), noise.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), training_config.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Post-step clear
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                del (
                    x,
                    cond_cam,
                    clean_cond_latents,
                    clean_gt_latent,
                    predicted_noise,
                    target_pred,
                    noise,
                )

            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                accelerator.log(logs, step=global_step)

        # End of epoch saving
        if accelerator.is_main_process:
            accelerator.save_state(
                os.path.join(training_config.output_dir, f"epoch_{epoch}")
            )


if __name__ == "__main__":
    main()
