#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import contextlib
from typing import Dict, Optional

import argbind
import torch
# from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import signal
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available, will use pytorch format")

from voxcpm.model import VoxCPMModel
from voxcpm.model.voxcpm import LoRAConfig
from voxcpm.training import (
    # Accelerator,
    BatchProcessor,
    TrainingTracker,
    # build_dataloader,
    load_audio_text_datasets,
    HFVoxCPMDataset,
)

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import json

logger = get_logger(__name__, log_level="INFO")

@argbind.bind(without_prefix=True)
def train(
    pretrained_path: str,
    train_manifest: str,
    val_manifest: str = "",
    sample_rate: int = 16_000,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    num_workers: int = 2,
    num_iters: int = 100_000,
    log_interval: int = 100,
    valid_interval: int = 1_000,
    save_interval: int = 10_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1_000,
    max_steps: int = 100_000,
    max_batch_tokens: int = 0,
    save_path: str = "checkpoints",
    tensorboard: str = "",
    lambdas: Dict[str, float] = {"loss/diff": 1.0, "loss/stop": 1.0},
    lora: dict = None,
    config_path: str = "",
    # Distribution options (for LoRA checkpoints)
    hf_model_id: str = "",   # HuggingFace model ID (e.g., "openbmb/VoxCPM1.5")
    distribute: bool = False, # If True, save hf_model_id as base_model; otherwise save pretrained_path
    project_name: str = "voxcpm-finetune",  # NEW: project name for Accelerate
    deepspeed_config: str = "",  # NEW: path to deepspeed config JSON
    seed: int = 42,              # NEW: random seed
):
    _ = config_path
    
    # Validate distribution options
    if lora is not None and distribute and not hf_model_id:
        raise ValueError("hf_model_id is required when distribute=True")
    
    # accelerator = Accelerator(amp=True)

    save_dir = Path(save_path)
    # tb_dir = Path(tensorboard) if tensorboard else save_dir / "logs"
    config = ProjectConfiguration(project_dir=save_path, logging_dir=str(Path(save_path) / "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps,
        mixed_precision="bf16",  # replaces amp=True
        project_config=config,
        log_with="tensorboard",
        deepspeed_plugin=deepspeed_config if deepspeed_config else None,
    )
    
    # Set seed for reproducibility
    set_seed(seed)

    if accelerator.is_main_process:  # Changed from accelerator.rank == 0
        save_dir.mkdir(parents=True, exist_ok=True)
        # tb_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()  # Changed from accelerator.barrier()

    # Initialize Accelerate's trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name, config={
            "lr": learning_rate, 
            "batch_size": batch_size,
            "grad_accum": grad_accum_steps
        })

    base_model = VoxCPMModel.from_local(pretrained_path, optimize=False, training=True, lora_config=LoRAConfig(**lora) if lora else None)
    tokenizer = base_model.text_tokenizer

    train_ds, val_ds = load_audio_text_datasets(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        sample_rate=sample_rate,
    )

    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    # Save original validation texts for audio generation display
    val_texts = None
    if val_ds is not None:
        val_texts = list(val_ds["text"])  # Save original texts
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    dataset_cnt = int(max(train_ds["dataset_id"])) + 1 if "dataset_id" in train_ds.column_names else 1
    num_train_samples = len(train_ds)

    # ------------------------------------------------------------------ #
    # Optional: filter samples by estimated token count to avoid OOM
    # Enabled when max_batch_tokens > 0:
    #   max_sample_len = max_batch_tokens // batch_size
    #   Samples exceeding this length will be dropped
    # ------------------------------------------------------------------ #
    if max_batch_tokens and max_batch_tokens > 0:
        from voxcpm.training.data import compute_sample_lengths

        audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
        est_lengths = compute_sample_lengths(
            train_ds,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        max_sample_len = max_batch_tokens // batch_size if batch_size > 0 else max(est_lengths)
        keep_indices = [i for i, L in enumerate(est_lengths) if L <= max_sample_len]

        if len(keep_indices) < len(train_ds) and accelerator.is_main_process:
            logger.info(
                f"Filtering {len(train_ds) - len(keep_indices)} / {len(train_ds)} "
                f"training samples longer than {max_sample_len} tokens "
                f"(max_batch_tokens={max_batch_tokens})."
            )
        train_ds = train_ds.select(keep_indices)

    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = (
        build_dataloader(
            val_ds,
            accelerator=accelerator,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_cnt,
        device=accelerator.device,
    )
    # Save audio_vae for audio generation
    audio_vae_for_gen = base_model.audio_vae
    del base_model.audio_vae

    optimizer = AdamW(
        (p for p in base_model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Cosine + warmup scheduler from transformers:
    # - num_warmup_steps: warmup steps
    # - num_training_steps: total training steps (outer step count)
    total_training_steps = max_steps if max_steps > 0 else num_iters
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    model, optimizer, scheduler, train_loader = accelerator.prepare(
        base_model, optimizer, scheduler, train_loader
    )
    
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    batch_processor.device = accelerator.device
    model.train()

    # Try to load checkpoint and resume training
    start_step = 0
    resume_dir = Path(save_path) / "latest_state"
    if resume_dir.exists():
        try:
            accelerator.load_state(str(resume_dir))
            logger.info(f"Loaded state from {resume_dir}")
            if accelerator.is_main_process:
                info_path = resume_dir / "custom_checkpoint_info.json"
                if info_path.exists():
                    with open(info_path, "r") as f:
                        start_step = json.load(f).get("step", 0)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    start_step_tensor = torch.tensor(start_step, device=accelerator.device)
    start_step_tensor = accelerator.reduce(start_step_tensor, reduction="sum")
    start_step = int(start_step_tensor.item())
    
    if start_step > 0 and accelerator.is_main_process:
        logger.info(f"Resuming training from step {start_step}")

    # Resume tracker for signal handler to read current step
    resume = {"step": start_step}

    # Register signal handler to save checkpoint on termination (SIGTERM/SIGINT)
    def _signal_handler(signum, frame, _model=model, _optim=optimizer, _sched=scheduler, _save_dir=save_dir, _pretrained=pretrained_path, _hf_id=hf_model_id, _dist=distribute, _resume=resume):
        try:
            cur_step = int(_resume.get("step", start_step))
        except Exception:
            cur_step = start_step
        print(f"Signal {signum} received. Saving checkpoint at step {cur_step} ...")
        try:
            save_checkpoint(_model, _optim, _sched, _save_dir, cur_step, _pretrained, _hf_id, _dist)
            print("Checkpoint saved. Exiting.")
        except Exception as e:
            print(f"Error saving checkpoint on signal: {e}")
        os._exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Manual epoch management instead of itertools.cycle to support DistributedSampler.set_epoch()
    grad_accum_steps = max(int(grad_accum_steps), 1)
    data_epoch = 0
    train_iter = iter(train_loader)

    def get_next_batch():
        """Get next batch, handles epoch boundary and DistributedSampler."""
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            # Key: set DistributedSampler epoch to ensure different data order each epoch
            sampler = getattr(train_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    for step in range(start_step, num_iters):
        # update resume step so signal handler can save current progress
        resume["step"] = step
        loss_dict = {}
        # Use accelerator's gradient accumulation context
        with accelerator.accumulate(model):
            batch = get_next_batch()
            processed = batch_processor(batch)

            # No need for manual sync context - handled by accelerator.accumulate()
            with accelerator.autocast():
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=step / max(1, num_iters),
                )

            total_loss = 0.0
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    weight = lambdas.get(key, 1.0)
                    # No manual division by grad_accum_steps - Accelerator handles this
                    loss_value = value * weight
                    total_loss = total_loss + loss_value
                    loss_dict[key] = value.detach()

            # Use accelerator's backward
            accelerator.backward(total_loss)

            # Clip gradients (DeepSpeed handles this internally if configured)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # if step % log_interval == 0 or step == num_iters - 1:
        if accelerator.sync_gradients and (step % log_interval == 0 or step == num_iters - 1):
            loss_values = {f"train/{k}": v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
            loss_values["train/lr"] = float(optimizer.param_groups[0]["lr"])
            # Approximate epoch: seen samples / total samples (considering grad_accum and batch_size)
            epoch = (step * grad_accum_steps * batch_size) / max(1, num_train_samples)
            loss_values["train/epoch"] = float(epoch)
            # loss_values["grad_norm"] = float(grad_norm)
            loss_values["train/grad_norm"] = float(grad_norm) if 'grad_norm' in locals() else 0.0
            accelerator.log(loss_values, step=step)

        if val_loader is not None and (step % valid_interval == 0 or step == num_iters - 1):
            validate(accelerator.unwrap_model(model), val_loader, batch_processor, accelerator, None, lambdas,
                    None, step=step, val_ds=val_ds, audio_vae=audio_vae_for_gen, 
                    sample_rate=sample_rate, val_texts=val_texts, tokenizer=tokenizer,
                    valid_interval=valid_interval)

        if step % save_interval == 0 or step == num_iters - 1:
            save_checkpoint(model, optimizer, scheduler, save_dir, step, pretrained_path, hf_model_id, distribute, accelerator)

    save_checkpoint(model, optimizer, scheduler, save_dir, num_iters, pretrained_path, hf_model_id, distribute, accelerator)
    accelerator.end_training()


def validate(model, val_loader, batch_processor, accelerator, tracker, lambdas, 
              writer=None, step=0, val_ds=None, audio_vae=None, sample_rate=22050,
              val_texts=None, tokenizer=None, valid_interval=1000):
    """Validate and generate sample audio"""
    import numpy as np
    from collections import defaultdict
    
    model.eval()
    total_losses = []
    sub_losses = defaultdict(list)  # Track individual sub-losses
    num_batches = 0
    max_val_batches = 10

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_val_batches:
                break
            processed = batch_processor(batch)
            outputs = model(
                processed["text_tokens"],
                processed["text_mask"],
                processed["audio_feats"],
                processed["audio_mask"],
                processed["loss_mask"],
                processed["position_ids"],
                processed["labels"],
                progress=0.0,
                sample_generate=False,
            )
            total = 0.0
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    weighted_loss = lambdas.get(key, 1.0) * value
                    total += weighted_loss
                    sub_losses[key].append(value.detach())
            total_losses.append(total.detach())
            num_batches += 1

    if total_losses:
        # Compute mean total loss
        mean_total_loss = torch.stack(total_losses).mean()
        # accelerator.all_reduce(mean_total_loss)
        mean_total_loss = accelerator.gather(mean_total_loss.unsqueeze(0)).mean()
        
        # Compute mean of each sub-loss
        val_metrics = {"val/total": mean_total_loss.item()}
        for key, values in sub_losses.items():
            mean_sub_loss = torch.stack(values).mean()
            # accelerator.all_reduce(mean_sub_loss)
            mean_sub_loss = accelerator.gather(mean_sub_loss.unsqueeze(0)).mean()
            val_metrics[key.replace("loss/", "val/")] = mean_sub_loss.item()
        
        accelerator.log(val_metrics, step=step)
    
    # Generate sample audio for TensorBoard display
    tb_tracker = accelerator.get_tracker("tensorboard")
    writer = tb_tracker.writer if tb_tracker else None
    if accelerator.is_main_process and val_ds is not None and audio_vae is not None and writer is not None:
        try:
            model.audio_vae = audio_vae.to(accelerator.device).float()
            generate_sample_audio(model, val_ds, audio_vae, writer, step, accelerator, sample_rate,
                                 val_texts=val_texts, tokenizer=tokenizer, valid_interval=valid_interval,
                                 tracker=tracker)
            model.audio_vae = None
        except Exception as e:
             logger.warning(f"Audio gen failed: {e}")
    
    model.train()


def compute_mel_spectrogram(audio_np, sample_rate, n_mels=128):
    """Compute Mel Spectrogram (dB) using librosa"""
    import numpy as np
    import librosa
    audio_np = audio_np.flatten().astype(np.float32)
    mel = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate, n_mels=n_mels, fmax=sample_rate // 2)
    return librosa.power_to_db(mel, ref=np.max)


def create_mel_figure(gen_audio_np, gen_mel, sample_rate, step=None, ref_audio_np=None, ref_mel=None):
    """
    Create mel spectrogram figure: show comparison if reference audio exists, otherwise show generated only
    """
    pass


def normalize_audio(audio_np):
    """Normalize audio to [-0.9, 0.9]"""
    import numpy as np
    max_val = np.abs(audio_np).max()
    return audio_np / max_val * 0.9 if max_val > 0 else audio_np


def generate_sample_audio(model, val_ds, audio_vae, writer, step, accelerator, sample_rate=22050, 
                          val_texts=None, tokenizer=None, pretrained_path=None, valid_interval=1000,
                          tracker=None):
    """Select 2 fixed validation samples, generate audio and log to TensorBoard"""
    import numpy as np
    
    # log = tracker.print if tracker else print
    num_samples = min(2, len(val_ds))
    logger.info(f"[Audio] Starting audio generation for {num_samples} samples at step {step}")
    
    for i in range(num_samples):
        sample = val_ds[i]
        text = val_texts[i] if val_texts and i < len(val_texts) else "Hello, this is a test."
        
        # Load reference audio
        ref_audio_np = None
        try:
            if "audio" in sample and isinstance(sample["audio"], dict) and "array" in sample["audio"]:
                ref_audio_np = np.array(sample["audio"]["array"], dtype=np.float32)
                ref_sr = sample["audio"].get("sampling_rate", sample_rate)
                if ref_sr != sample_rate:
                    import torchaudio.functional as F
                    ref_audio_np = F.resample(torch.from_numpy(ref_audio_np).unsqueeze(0), ref_sr, sample_rate).squeeze(0).numpy()
                logger.info(f"[Audio] Loaded reference audio for sample {i}: duration={len(ref_audio_np)/sample_rate:.2f}s")
        except Exception as e:
            logger.warning(f"[Warning] Failed to load reference audio: {e}")
        
        try:
            logger.info(f"[Audio] Generating sample {i} with text: '{text[:50]}...'")
            with torch.no_grad():
                generated = model.generate(target_text=text, inference_timesteps=10, cfg_value=2.0)
            
            if generated is None or len(generated) == 0:
                logger.warning(f"[Warning] Generated audio is empty for sample {i}")
                continue
            
            # Process generated audio
            gen_audio_np = generated.cpu().float().numpy().flatten() if isinstance(generated, torch.Tensor) else np.array(generated, dtype=np.float32).flatten()
            gen_audio_np = normalize_audio(gen_audio_np)
            
            tag = f"val_sample_{i}"
            writer.add_audio(f"{tag}/generated_audio", gen_audio_np, global_step=step, sample_rate=sample_rate)
            logger.info(f"[Audio] Generated audio for sample {i}: duration={len(gen_audio_np)/sample_rate:.2f}s")
            
            # Log reference audio
            if ref_audio_np is not None:
                writer.add_audio(f"{tag}/reference_audio", normalize_audio(ref_audio_np), global_step=step, sample_rate=sample_rate)
                
        except Exception as e:
            logger.warning(f"[Warning] Failed to generate audio for sample {i}: {e}")


def load_checkpoint(model, optimizer, scheduler, save_dir: Path, accelerator):
    """
    Load the latest checkpoint if it exists.
    Returns the step number to resume from, or 0 if no checkpoint found.
    """
    pass


def save_checkpoint(model, optimizer, scheduler, save_dir: Path, step: int, pretrained_path: str = None, hf_model_id: str = "", distribute: bool = False, accelerator=None):
    """
    Save checkpoint with different strategies for full finetune vs LoRA:
    - Full finetune: save non-vae weights to model.safetensors (or pytorch_model.bin if safetensors unavailable)
    - LoRA: save only lora weights to lora_weights.safetensors (or lora_weights.ckpt if safetensors unavailable)
    """
    import shutil
    
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = "latest_state" if step == 0 else f"step_{step:07d}"
    folder = save_dir / tag
    folder.mkdir(parents=True, exist_ok=True)
    
    # unwrapped = model.module if hasattr(model, "module") else model
    # full_state = unwrapped.state_dict()
    accelerator.save_state(str(folder))

    # Save custom step info
    if accelerator.is_main_process:
        step_info = {"step": step}
        with open(save_dir / "latest_state" / "custom_checkpoint_info.json", "w") as f:
            json.dump(step_info, f)
    # Unwrap model to save additional metadata
    unwrapped = accelerator.unwrap_model(model)
    lora_cfg = unwrapped.lora_config
    
    if lora_cfg is not None:
        # # LoRA finetune: save only lora_A/lora_B weights
        # state_dict = {k: v for k, v in full_state.items() if "lora_" in k}
        # if SAFETENSORS_AVAILABLE:
        #     save_file(state_dict, folder / "lora_weights.safetensors")
        # else:
        #     torch.save({"state_dict": state_dict}, folder / "lora_weights.ckpt")
        
        # Save LoRA config and base model path to a separate JSON file
        # If distribute=True, save hf_model_id; otherwise save local pretrained_path
        import json
        base_model_to_save = hf_model_id if distribute else (str(pretrained_path) if pretrained_path else None)
        lora_info = {
            "base_model": base_model_to_save,
            "lora_config": lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg),
        }
        with open(folder / "lora_config.json", "w", encoding="utf-8") as f:
            json.dump(lora_info, f, indent=2, ensure_ascii=False)
    else:
        # # Full finetune: save non-vae weights to model.safetensors
        # state_dict = {k: v for k, v in full_state.items() if not k.startswith("audio_vae.")}
        # if SAFETENSORS_AVAILABLE:
        #     save_file(state_dict, folder / "model.safetensors")
        # else:
        #     torch.save({"state_dict": state_dict}, folder / "pytorch_model.bin")
        
        # Copy config files from pretrained path
        if pretrained_path:
            pretrained_dir = Path(pretrained_path)
            files_to_copy = ["config.json", "audiovae.pth", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json"]
            for fname in files_to_copy:
                src = pretrained_dir / fname
                if src.exists():
                    shutil.copy2(src, folder / fname)

    # Update (or create) a `latest` symlink pointing to the most recent checkpoint folder
    latest_link = save_dir / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            # remove existing link or directory
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        # Create a symlink pointing to the new folder
        os.symlink(str(folder), str(latest_link))
    except Exception:
        # If symlink creation fails (e.g., on Windows or permission issues), fall back to copying
        try:
            if latest_link.exists():
                if latest_link.is_dir():
                    shutil.rmtree(latest_link)
                else:
                    latest_link.unlink()
            shutil.copytree(folder, latest_link)
        except Exception:
            print(f"Warning: failed to update latest checkpoint link at {latest_link}")

from datasets import Audio, Dataset
def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVoxCPMDataset(hf_dataset)
    # Standard padding-based batching; Accelerator will attach DistributedSampler if needed.

    return torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=HFVoxCPMDataset.collate_fn,
        drop_last=drop_last,
        pin_memory=True,
    )

if __name__ == "__main__":
    from voxcpm.training.config import load_yaml_config

    args = argbind.parse_args()
    config_file = args.get("config_path")
    # If YAML config provided, use YAML args to call train
    if config_file:
        yaml_args = load_yaml_config(config_file)
        train(**yaml_args)
    else:
        # Otherwise use command line args (parsed by argbind)
        with argbind.scope(args):
            train()