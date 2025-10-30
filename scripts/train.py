
import os
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
from src.models.cog_vae.vae_cogvideox import VAE
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
    num_workers = 4 # For DataLoader

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
    ckpt_path = "/content/local_checkpoints/ckpt/ema.pt" # Path to pre-trained weights in local Colab storage
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

                # Move VAE to GPU for encoding, then back to CPU
                vae.to(accelerator.device)
                with torch.no_grad():
                    # Placeholder: Assuming batch["images_gt"] will contain ground truth images
                    # This needs to be implemented in src/data/dataset.py
                    # For now, we'll use a dummy tensor or adapt to existing output
                    # If dataset doesn't return GT, this will crash. We'll fix in Day 2.
                    dummy_gt_images = torch.randn(1, 3, 256, 256).to(accelerator.device) # Placeholder
                    latents = vae.encode(dummy_gt_images).latents # Placeholder
                vae.to("cpu")

                # Predict noise (simplified for now, will use scheduler in Day 2/3)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=accelerator.device)
                noisy_latents = latents + noise # Simplified noise addition for now
                
                # Forward pass
                # Placeholder: model_args needs to be adapted from inference_utils.py
                # For now, we'll pass minimal args
                # This will likely need adjustment once dataset returns proper conditioning
                dummy_cond_cam = torch.randn(1, 1, 3, 256, 256).to(accelerator.device)
                dummy_bbox = {"bboxes": torch.randn(1, 1, 10, 8, 3).to(accelerator.device), "classes": torch.randint(0, 8, (1, 1, 10)).to(accelerator.device), "masks": torch.ones(1, 1, 10).to(accelerator.device)}
                dummy_cams = torch.randn(1, 1, 7, 3, 7).to(accelerator.device)
                dummy_height = torch.tensor([256]).to(accelerator.device)
                dummy_width = torch.tensor([256]).to(accelerator.device)
                dummy_NC = 5 # Number of cameras

                predicted_noise = model(noisy_latents, timesteps, 
                                        cond_cam=dummy_cond_cam, 
                                        bbox=dummy_bbox, 
                                        cams=dummy_cams, 
                                        height=dummy_height, 
                                        width=dummy_width, 
                                        NC=dummy_NC)

                # Calculate loss (MSE for now)
                loss = F.mse_mse_loss(predicted_noise, noise, reduction="mean")
                
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
