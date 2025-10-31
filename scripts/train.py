
import os
import sys
# Add project root to sys.path to enable imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, cpu_offload
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import pathlib
from einops import repeat

torch.backends.cudnn.benchmark = False
torch.backends.cuda.enable_flash_sdp(False)  # Standard attn, less VRAM

def tile_vae_encode(vae, video, tile_size=128, overlap=16):

    B, C, T, H, W = video.shape

    step = tile_size - overlap

    tiles = []

    tile_positions = []  # (row_start_latent, col_start_latent)



    # Extract tiles

    for row in range(0, H, step):

        for col in range(0, W, step):

            end_row = min(row + tile_size, H)

            end_col = min(col + tile_size, W)

            tile = video[:, :, :, row:end_row, col:end_col]

            pad_h = tile_size - tile.shape[3]

            pad_w = tile_size - tile.shape[4]

            tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='constant', value=0)

            with torch.no_grad():

                latent_tile = vae.encode(tile)

            actual_h = (end_row - row) // 8

            actual_w = (end_col - col) // 8

            latent_tile = latent_tile[:, :, :, :actual_h, :actual_w]

            tiles.append(latent_tile)

            tile_positions.append((row // 8, col // 8))

            del tile, latent_tile

            torch.cuda.empty_cache()

            gc.collect()



    # Stitch: Iterate over actual tiles (dynamic, no grid)

    latent_H = H // 8

    latent_W = W // 8

    stitched = torch.zeros(B, tiles[0].shape[1], T//4, latent_H, latent_W, device=video.device)

    count = torch.zeros(B, 1, T//4, latent_H, latent_W, device=video.device)



    for tile_idx, (row_start, col_start) in enumerate(tile_positions):

        latent_tile = tiles[tile_idx]

        row_end = min(row_start + latent_tile.shape[3], latent_H)

        col_end = min(col_start + latent_tile.shape[4], latent_W)

        h_slice = slice(row_start, row_end)

        w_slice = slice(col_start, col_end)

        stitched[:, :, :, h_slice, w_slice] += latent_tile[:, :, :, :row_end-row_start, :col_end-col_start]

        count[:, :, :, h_slice, w_slice] += 1



    stitched /= count.clamp(min=1)

    del tiles

    torch.cuda.empty_cache()

    gc.collect()

    return stitched



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

    mixed_precision = 'bf16'  # Stable for softmax, no fp32 cast

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

    torch.backends.cuda.enable_mem_efficient_sdp(False)

    

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

                # Ultra OOM Guard

                torch.cuda.empty_cache()

                gc.collect()

                torch.cuda.synchronize()



                # === DUMMY DATA (Full Res) ===

                B, T, C, H, W = 1, 4, 3, 256, 256  # latent_T=1, H/W=32

                NC = 5

                N_COND = NC - 1

                device = accelerator.device

                dummy_gt_video = torch.randn(B, C, T, H, W, device=device)

                dummy_cond_cam_raw = torch.randn(B, N_COND, C, T, H, W, device=device)



                # === 3. Encode clean latents (Tiled, Fixed) ===

                torch.cuda.empty_cache()

                gc.collect()

                vae.to(device)

                with torch.no_grad():

                    clean_gt_latent = tile_vae_encode(vae, dummy_gt_video)

                    # Pad to exact (..., 1, 32, 32)

                    if clean_gt_latent.shape[2] < 1:

                        clean_gt_latent = F.pad(clean_gt_latent, (0, 0, 0, 1 - clean_gt_latent.shape[2]), mode='constant', value=0)

                    if clean_gt_latent.shape[3] < 32:

                        clean_gt_latent = F.pad(clean_gt_latent, (0, 32 - clean_gt_latent.shape[4], 0, 32 - clean_gt_latent.shape[3]), mode='constant', value=0)

                    B_gt, C_latent, latent_T, latent_H, latent_W = clean_gt_latent.shape



                    cond_reshaped = dummy_cond_cam_raw.view(B * N_COND, C, T, H, W)

                    cond_latents_tiled = []

                    for i in range(B * N_COND):

                        cond_tile = tile_vae_encode(vae, cond_reshaped[i:i+1])

                        # Pad to exact (..., 1, 32, 32)

                        if cond_tile.shape[2] < 1:

                            cond_tile = F.pad(cond_tile, (0, 0, 0, 1 - cond_tile.shape[2]), mode='constant', value=0)

                        if cond_tile.shape[3] < 32:

                            cond_tile = F.pad(cond_tile, (0, 32 - cond_tile.shape[4], 0, 32 - cond_tile.shape[3]), mode='constant', value=0)

                        cond_latents_tiled.append(cond_tile)

                        torch.cuda.empty_cache()

                        gc.collect()

                    cond_latents_raw = torch.cat(cond_latents_tiled, dim=0)

                    del cond_latents_tiled

                    cond_latents_raw = cond_latents_raw.view(B, N_COND, C_latent, latent_T, latent_H, latent_W)

                    clean_cond_latents = cond_latents_raw.view(B, N_COND * C_latent, latent_T, latent_H, latent_W)

                    del cond_latents_raw



                vae.to("cpu")

                torch.cuda.empty_cache()

                gc.collect()

                torch.cuda.synchronize()



                # === 4. Add noise ===

                noise = torch.randn_like(clean_gt_latent)

                noisy_target = clean_gt_latent + noise

                timesteps = torch.randint(0, 1000, (B,), device=device).long()



                x = clean_cond_latents

                cond_cam = noisy_target



                # === 6. Dummy conditioning (Minimal) ===

                num_objects = 1

                base_bbox = {

                    "bboxes": torch.randn(B, 1, num_objects, 8, 3, device=device),

                    "classes": torch.randint(0, 8, (B, 1, num_objects), device=device),

                    "masks": torch.ones(B, 1, num_objects, device=device)

                }

                base_cams = torch.randn(B, 1, 7, 3, 7, device=device)

                dummy_height = torch.tensor([H], device=device)

                dummy_width = torch.tensor([W], device=device)



                expanded_bbox = {k: repeat(v, "b ... -> (b nc) ...", nc=NC) for k, v in base_bbox.items()}

                expanded_cams = repeat(base_cams, "b ... -> (b nc) ...", nc=NC)



                # Pre-forward clear

                torch.cuda.empty_cache()

                gc.collect()

                torch.cuda.synchronize()



                # === 7. Forward ===

                predicted_noise = model(

                    x, timesteps,

                    cond_cam=cond_cam,

                    bbox=expanded_bbox,

                    cams=expanded_cams,

                    height=dummy_height,

                    width=dummy_width,

                    NC=NC

                )



                # === 8. Loss ===

                target_start = 3 * C_latent

                target_end = 4 * C_latent

                target_pred = predicted_noise[:, target_start:target_end, :, :, :]

                loss = F.mse_loss(target_pred, noise, reduction="mean")

                print(f"Loss: {loss.item():.4f}")



                accelerator.backward(loss)

                if accelerator.sync_gradients:

                    accelerator.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)



                optimizer.step()

                lr_scheduler.step()

                optimizer.zero_grad()



                # Post-step clear

                torch.cuda.empty_cache()

                gc.collect()

                torch.cuda.synchronize()

                del x, cond_cam, clean_cond_latents, clean_gt_latent, predicted_noise, target_pred, noise



            # Logging

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
