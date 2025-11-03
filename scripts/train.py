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





import math
import torch.nn as nn

def partial_load_checkpoint(model, checkpoint_path, skip_prefixes=['x_embedder', 'final_layer'], init_skipped=True):
    """
    Loads matching keys from checkpoint; skips mismatches (e.g., channel changes).
    Inits skipped layers randomly (kaiming) or zeros.
    Returns: loaded_count, skipped_keys
    """
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
            if 'spatial_blocks' in key:
                print(f"[SUCCESS] Loaded core layer: {key[:70]}... (shape {ckpt_value.shape})")
            else:
                print(f"[INFO] Loaded other layer: {key[:70]}... (shape {ckpt_value.shape})")
        else:
            skipped_keys.append(key)
            print(f"[WARNING] Shape mismatch (skipping): {key} – CKPT {ckpt_value.shape} vs MODEL {model_value.shape}")
            if init_skipped:
                if 'weight' in key:
                    print(f"  -> Initializing weight with Kaiming Uniform: {key}")
                    nn.init.kaiming_uniform_(model_value, a=math.sqrt(5))
                else:
                    print(f"  -> Initializing bias with Zeros: {key}")
                    nn.init.zeros_(model_value)

    model.load_state_dict(model_dict)
    print("\n--- Load Summary ---")
    print(f"Total keys in checkpoint: {len(state_dict)}")
    print(f"Total keys in model: {len(model_dict)}")
    print(f"Successfully loaded {loaded_count} matching keys.")
    print(f"Skipped {len(skipped_keys)} mismatched keys: {skipped_keys}")
    
    core_preserved = all('spatial_blocks' not in k for k in skipped_keys)
    print(f"Core `spatial_blocks` preserved: {core_preserved}")
    print("=================================")
    
    if not core_preserved:
        raise RuntimeError("Critical Error: Core spatial blocks were not loaded. Aborting.")
        
    return loaded_count, skipped_keys

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
    num_epochs = 10
    batch_size = 1  # Keep at 1 for now due to VRAM constraints
    gradient_accumulation_steps = 1  # Reduced to 1 to minimize memory
    learning_rate = 1e-5
    lr_warmup_steps = 500
    mixed_precision = "fp16"  # Use fp16 for the proof run
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
    ][:50] # Use only 50 scenes for the proof run
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
    model_config_dict['in_channels'] = 80
    if accelerator.is_main_process:
        print(f"[Model Build] Overriding model config. New config: {model_config_dict}")
    model = build_module(model_config_dict, MODELS)

    # Load pre-trained weights for STDiT3 using the custom partial loader
    ckpt_path = "/content/1380-solution_github/checkpoints/ckpt/ema.pt"
    loaded_count, skipped_keys = partial_load_checkpoint(model, ckpt_path)

    if len(skipped_keys) > 50:
        raise ValueError("Over 50 mismatched keys, something is wrong with the checkpoint or model architecture.")



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
                    images_gt = batch['images_gt']
                    if accelerator.is_main_process:
                        print(f"[VAE Encode] Input images_gt shape: {images_gt.shape}")

                    latents_list = []
                    for i in range(images_gt.size(1)):
                        view = images_gt[:, i:i+1, :, :, :]
                        view_permed = view.permute(0, 2, 1, 3, 4)
                        latent_view = vae.encode(view_permed)
                        latents_list.append(latent_view)
                    
                    gt_latents = torch.cat(latents_list, dim=1)
                    if accelerator.is_main_process:
                        print(f"[VAE Encode] Output gt_latents shape: {gt_latents.shape}")

                    vae.to("cpu")

                t = torch.rand(gt_latents.shape[0], device=accelerator.device).view(-1, 1, 1, 1, 1)
                noise = torch.randn_like(gt_latents)
                noisy_latents = (1 - t) * gt_latents + t * noise
                timesteps = t.squeeze(-1).squeeze(-1).squeeze(-1)

                from src.models.stdit.control_embedder import ControlEmbedder
                control_embedder_instance = ControlEmbedder(model.config)
                control_embedder_instance.to(accelerator.device)

                bboxes_list = batch['bboxes_3d_data']
                cond_emb = control_embedder_instance(
                    bboxes_list,
                    batch['camera_param'],
                    batch['bev_grid']
                )

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

                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                del predicted_noise, noise, cond_emb, gt_latents, noisy_latents

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

            # --- Rolling Checkpoint Management ---
            checkpoint_dirs = sorted(
                [d for d in os.listdir(training_config.output_dir) if d.startswith("epoch_")],
                key=lambda x: int(x.split('_')[-1])
            )
            max_checkpoints = 2 # Keep the 2 most recent checkpoints
            if len(checkpoint_dirs) > max_checkpoints:
                dir_to_delete = os.path.join(training_config.output_dir, checkpoint_dirs[0])
                print(f"[Checkpoint Manager] Deleting oldest checkpoint to save space: {dir_to_delete}")
                # Use shutil.rmtree for safely deleting directories
                import shutil
                shutil.rmtree(dir_to_delete)



if __name__ == "__main__":
    main()
