#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import contextlib
from typing import Dict, Optional

import argbind
import torch
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

from voxcpm.model import VoxCPMModel, VoxCPM2Model
from voxcpm.model.voxcpm import LoRAConfig as LoRAConfigV1
from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2
from voxcpm.training import (
    BatchProcessor,
    TrainingTracker,
    load_audio_text_datasets,
    HFVoxCPMDataset,
)

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import json
import gc
import wandb
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__, log_level="INFO")

class MaxTimeCallback:
    def __init__(self, max_time_seconds):
        self.max_time_seconds = max_time_seconds
        self.start_time = None
    
    def on_train_begin(self):
        self.start_time = time.time()
    
    def should_stop(self):
        return time.time() - self.start_time > self.max_time_seconds
    
@argbind.bind(without_prefix=True)
def train(
    pretrained_path: str,
    train_manifest: str,
    val_manifest: str = "",
    sample_rate: int = 16_000,
    out_sample_rate: int = 0,    # AudioVAE decoder output rate; used for TensorBoard audio logging
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    num_workers: int = 2,
    log_interval: int = 100,
    valid_interval: int = 1_000,
    save_interval: int = 10_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1_000,
    max_steps: int = 100_000,
    max_batch_tokens: int = 0,
    save_path: str = "checkpoints",
    log_with: str = "wandb",
    wandb_name = "voxcpm",
    lambdas: Dict[str, float] = {"loss/diff": 1.0, "loss/stop": 1.0},
    lora: dict = None,
    config_path: str = "",
    max_grad_norm: float = 0.0,  # gradient clipping; 0 = disabled
    # Distribution options (for LoRA checkpoints)
    hf_model_id: str = "",       # HuggingFace model ID (e.g., "openbmb/VoxCPM1.5")
    distribute: bool = False,    # If True, save hf_model_id as base_model; otherwise save pretrained_path
    project_name: str = "voxcpm-finetune",  
    deepspeed_config: str = "",  
    seed: int = 42,              
    max_time_seconds: int = 0,   
    resume_dir: str = "",        
):
    _ = config_path
    time_callback = MaxTimeCallback(max_time_seconds)
    time_callback.on_train_begin()
    
    # Validate distribution options
    if lora is not None and distribute and not hf_model_id:
        raise ValueError("hf_model_id is required when distribute=True")

    save_dir = Path(save_path)
    config = ProjectConfiguration(project_dir=save_path, logging_dir=str(Path(save_path) / "logs"))
    ds_plugin = None
    if deepspeed_config and deepspeed_config.strip():
        ds_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps,
        mixed_precision="bf16",  
        project_config=config,
        log_with=["tensorboard","wandb"] if log_with == "wandb" else["tensorboard"],
        deepspeed_plugin=ds_plugin if ds_plugin else None,
    )
    
    # Set seed for reproducibility
    set_seed(seed)

    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # Initialize Accelerate's trackers
    if accelerator.is_main_process:
        init_kwargs={
            "wandb": {
                "name": wandb_name,
                "settings": wandb.Settings(_disable_stats=True)
            }
        } if log_with == "wandb" else {}
        accelerator.init_trackers(project_name, config={
            "lr": learning_rate, 
            "batch_size": batch_size,
            "grad_accum": grad_accum_steps
            },
            init_kwargs=init_kwargs
        )

    # Auto-detect model architecture from config.json
    with open(os.path.join(pretrained_path, "config.json"), "r", encoding="utf-8") as _f:
        _arch = json.load(_f).get("architecture", "voxcpm").lower()
    _model_cls = VoxCPM2Model if _arch == "voxcpm2" else VoxCPMModel
    LoRAConfig = LoRAConfigV2 if _arch == "voxcpm2" else LoRAConfigV1
    
    if accelerator.is_main_process:
        logger.info(f"Detected architecture: {_arch} -> {_model_cls.__name__}")

    base_model = _model_cls.from_local(
        pretrained_path, optimize=False, training=True, lora_config=LoRAConfig(**lora) if lora else None
    )
    tokenizer = base_model.text_tokenizer
    
    expected_sr = getattr(base_model.audio_vae, "sample_rate", sample_rate)
    assert sample_rate == expected_sr, (
        f"sample_rate mismatch: config says {sample_rate}, but the AudioVAE encoder expects {expected_sr}. "
        f"Please set sample_rate: {expected_sr} in your training config. "
    )

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        apply_activation_checkpointing,
        CheckpointImpl
    )

    # Wrap individual layers
    def apply_gradient_checkpointing(model):
        """Apply checkpointing using PyTorch's distributed checkpoint_wrapper"""
        patched_count = 0
        
        for name, module in model.named_modules():
            if hasattr(module, "layers") and isinstance(module.layers, torch.nn.ModuleList):
                for i, layer in enumerate(module.layers):
                    if not hasattr(layer, "_is_checkpointed"):
                        # Wrap the entire layer module
                        module.layers[i] = checkpoint_wrapper(
                            layer,
                            checkpoint_impl=CheckpointImpl.NO_REENTRANT
                        )
                        module.layers[i]._is_checkpointed = True
                        patched_count += 1
        
        return patched_count

    num_patched = apply_gradient_checkpointing(base_model)
    logger.info(f"Successfully patched {num_patched} layers for gradient checkpointing.")
    if hasattr(base_model, "fsq_layer"):
        base_model.fsq_layer = base_model.fsq_layer.float()
    if hasattr(base_model, "stop_loss"):
        base_model.stop_loss = base_model.stop_loss.float()

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

    start_time = time.time()
    if max_batch_tokens and max_batch_tokens > 0:
        from voxcpm.training.data import compute_sample_lengths

        audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
        est_lengths = compute_sample_lengths(
            train_ds,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        max_sample_len = max_batch_tokens // batch_size if batch_size > 0 else max(est_lengths)
        keep_indices =[i for i, L in enumerate(est_lengths) if L <= max_sample_len]

        if len(keep_indices) < len(train_ds) and accelerator.is_main_process:
            logger.info(
                f"Filtering {len(train_ds) - len(keep_indices)} / {len(train_ds)} "
                f"training samples longer than {max_sample_len} tokens "
                f"(max_batch_tokens={max_batch_tokens})."
            )
        train_ds = train_ds.select(keep_indices)
        end_time = time.time()
        logger.info(f"Filtering time: {end_time - start_time:.2f} seconds")

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
    
    # Save audio_vae and output sample rate for audio generation.
    out_sr = getattr(base_model, "sample_rate", 0)  # decoder output rate
    if out_sr == 0 and out_sample_rate > 0:
        out_sr = out_sample_rate

    if hasattr(base_model, "audio_vae") and base_model.audio_vae is not None:
        logger.info("Offloading Audio VAE to CPU...")
        audio_vae_for_gen = base_model.audio_vae
        # Freeze VAE parameters
        for p in audio_vae_for_gen.parameters():
            p.requires_grad = False
        # Detach from model to prevent DDP broadcasting
        del base_model.audio_vae
        base_model.audio_vae = None 
    else:
        audio_vae_for_gen = None

    # Clean up before optimizer init
    gc.collect()
    torch.cuda.empty_cache()

    import bitsandbytes as bnb
    logger.info("Using 8-bit AdamW optimizer.")
    optimizer = bnb.optim.AdamW8bit(
        (p for p in base_model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
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
    resume_dir = Path(resume_dir) if resume_dir else None
    if resume_dir and resume_dir.exists():
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

    resume = {"step": start_step}

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
            sampler = getattr(train_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    for step in range(start_step, max_steps):
        resume["step"] = step
        loss_dict = {}
        # Use accelerator's gradient accumulation context
        with accelerator.accumulate(model):
            batch = get_next_batch()
            processed = batch_processor(batch)

            outputs = model(
                processed["text_tokens"],
                processed["text_mask"],
                processed["audio_feats"],
                processed["audio_mask"],
                processed["loss_mask"],
                processed["position_ids"],
                processed["labels"],
                progress=step / max(1, max_steps),
            )

            total_loss = 0.0
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    weight = lambdas.get(key, 1.0)
                    loss_value = value * weight
                    total_loss = total_loss + loss_value
                    loss_dict[key] = value.detach()

            accelerator.backward(total_loss)

            if accelerator.sync_gradients:
                effective_max_norm = max_grad_norm if max_grad_norm > 0 else 1.0
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=effective_max_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        should_stop_val = 0.0
        if step % log_interval == 0 or step == max_steps - 1:
            should_stop = torch.tensor(0.0, device=accelerator.device)
            if time_callback.should_stop():
                should_stop += 1.0
            accelerator.reduce(should_stop, reduction="sum")
            should_stop_val = should_stop.item()

            loss_values = {f"train/{k}": v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
            loss_values["train/lr"] = float(optimizer.param_groups[0]["lr"])
            epoch = (step * grad_accum_steps * batch_size * accelerator.num_processes) / max(1, num_train_samples)
            loss_values["train/epoch"] = float(epoch)
            loss_values["train/grad_norm"] = float(grad_norm) if 'grad_norm' in locals() else 0.0
            accelerator.log(loss_values, step=step)

        if should_stop_val > 0:
            if accelerator.is_main_process:
                global_batch_size = batch_size * grad_accum_steps * accelerator.num_processes
                total_samples_seen = (step + 1) * global_batch_size
                total_epochs = total_samples_seen / num_train_samples
                
                logger.info(f"Training stopped due to max_time_seconds limit.")
                logger.info(f"--------------------------------------------------")
                logger.info(f"Global Step:       {step}")
                logger.info(f"Epochs Completed:  {total_epochs:.2f}")
                logger.info(f"--------------------------------------------------")
            break

        if val_loader is not None and (step % valid_interval == 0 or step == max_steps - 1):
            gc.collect()
            torch.cuda.empty_cache()
            validate(model, val_loader, batch_processor, accelerator, None, lambdas,
                    writer=None, step=step, val_ds=val_ds, audio_vae=audio_vae_for_gen, 
                    sample_rate=sample_rate, out_sample_rate=out_sr, 
                    val_texts=val_texts, tokenizer=tokenizer, valid_interval=valid_interval)
            gc.collect()
            torch.cuda.empty_cache()

        if step % save_interval == 0 and step > start_step:
            save_checkpoint(model, optimizer, scheduler, save_dir, step, pretrained_path, hf_model_id, distribute, accelerator)

    save_checkpoint(model, optimizer, scheduler, save_dir, step+1, pretrained_path, hf_model_id, distribute, accelerator)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Ending training and closing trackers...")
    try:
        accelerator.end_training()
    except Exception as e:
        if accelerator.is_main_process:
            logger.warning(f"Exception during end_training (usually safe to ignore): {e}")
        if hasattr(accelerator, '_trackers'):
            for tracker in accelerator._trackers:
                try:
                    if hasattr(tracker, 'finish'):
                        tracker.finish()
                except:
                    pass

def validate(model, val_loader, batch_processor, accelerator, tracker, lambdas, 
              writer=None, step=0, val_ds=None, audio_vae=None, sample_rate=22050,
              out_sample_rate=0, val_texts=None, tokenizer=None, valid_interval=1000):
    """Validate and generate sample audio"""
    import numpy as np
    from collections import defaultdict
    
    model.eval()
    total_losses =[]
    sub_losses = defaultdict(list)
    num_batches = 0
    max_val_batches = 10

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_val_batches:
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
            
    accelerator.wait_for_everyone()
    
    if total_losses:
        mean_total_loss = torch.stack(total_losses).mean()
        mean_total_loss = accelerator.gather(mean_total_loss.unsqueeze(0)).mean()
        
        val_metrics = {"val/total": mean_total_loss.item()}
        for key, values in sub_losses.items():
            mean_sub_loss = torch.stack(values).mean()
            mean_sub_loss = accelerator.gather(mean_sub_loss.unsqueeze(0)).mean()
            val_metrics[key.replace("loss/", "val/")] = mean_sub_loss.item()
        
        accelerator.log(val_metrics, step=step)
    
    # Generate sample audio for TensorBoard display
    if accelerator.is_main_process and val_ds is not None and audio_vae is not None:
        tb_tracker = accelerator.get_tracker("tensorboard")
        if tb_tracker is not None:
            writer = tb_tracker.writer
            try:
                unwrapped_model = accelerator.unwrap_model(model)
                generate_sample_audio(unwrapped_model, val_ds, audio_vae, writer, step, accelerator, 
                                      sample_rate, out_sample_rate=out_sample_rate,
                                      val_texts=val_texts, tokenizer=tokenizer, 
                                      valid_interval=valid_interval, tracker=tracker)
            except Exception as e:
                logger.warning(f"Audio generation failed: {e}")
                import traceback
                logger.warning(traceback.format_exc())
            finally:
                if hasattr(unwrapped_model, "audio_vae"):
                    unwrapped_model.audio_vae = None
                    
    accelerator.wait_for_everyone()
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    fmax = sample_rate // 2
    step_str = f" @ Step {step}" if step is not None else ""

    if ref_audio_np is not None and ref_mel is not None:
        # Comparison mode: reference vs generated
        fig, (ax_ref, ax_gen) = plt.subplots(2, 1, figsize=(12, 8))

        img_ref = librosa.display.specshow(
            ref_mel, sr=sample_rate, x_axis="time", y_axis="mel", fmax=fmax, cmap="viridis", ax=ax_ref
        )
        ax_ref.set_title(
            f"Reference (GT) - {len(ref_audio_np)/sample_rate:.2f}s{step_str}",
            fontsize=10,
            fontweight="bold",
            color="#28A745",
        )
        plt.colorbar(img_ref, ax=ax_ref, format="%+2.0f dB", pad=0.02)

        img_gen = librosa.display.specshow(
            gen_mel, sr=sample_rate, x_axis="time", y_axis="mel", fmax=fmax, cmap="viridis", ax=ax_gen
        )
        ax_gen.set_title(
            f"Generated - {len(gen_audio_np)/sample_rate:.2f}s", fontsize=10, fontweight="bold", color="#DC3545"
        )
        plt.colorbar(img_gen, ax=ax_gen, format="%+2.0f dB", pad=0.02)
    else:
        # Single figure mode: show generated only
        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            gen_mel, sr=sample_rate, x_axis="time", y_axis="mel", fmax=fmax, cmap="viridis", ax=ax
        )
        ax.set_title(f"Generated - {len(gen_audio_np)/sample_rate:.2f}s{step_str}", fontsize=11, fontweight="bold")
        plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)

    plt.tight_layout()
    return fig


def normalize_audio(audio_np):
    """Normalize audio to [-0.9, 0.9]"""
    import numpy as np
    max_val = np.abs(audio_np).max()
    return audio_np / max_val * 0.9 if max_val > 0 else audio_np


def generate_sample_audio(model, val_ds, audio_vae, writer, step, accelerator, sample_rate=22050, 
                          out_sample_rate=0, val_texts=None, tokenizer=None, pretrained_path=None, 
                          valid_interval=1000, tracker=None):
    """Select 2 fixed validation samples, generate audio and log to TensorBoard"""
    import numpy as np
    
    log = logger.info
    num_samples = min(2, len(val_ds))
    log(f"[Audio] Starting audio generation for {num_samples} samples at step {step}")
    
    gen_sr = out_sample_rate if out_sample_rate > 0 else sample_rate
    
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
                log(f"[Audio] Loaded reference audio for sample {i}: duration={len(ref_audio_np)/sample_rate:.2f}s")
        except Exception as e:
            logger.warning(f"[Warning] Failed to load reference audio: {e}")
            
        prev_training = model.training
        try:
            model.eval()
            model.audio_vae = audio_vae.to(accelerator.device).to(torch.float32)
            
            log(f"[Audio] Generating sample {i} with text: '{text[:50]}...'")
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            )
            with torch.no_grad():
                with autocast_ctx:
                    generated = model.generate(target_text=text, inference_timesteps=10, cfg_value=2.0)
            
            if generated is None or len(generated) == 0:
                logger.warning(f"[Warning] Generated audio is empty for sample {i}")
                continue
            
            # Process generated audio
            gen_audio_np = generated.cpu().float().numpy().flatten() if isinstance(generated, torch.Tensor) else np.array(generated, dtype=np.float32).flatten()
            gen_audio_np = normalize_audio(gen_audio_np)
            
            tag = f"val_sample_{i}"
            if writer:
                writer.add_audio(f"{tag}/generated_audio", gen_audio_np, global_step=step, sample_rate=gen_sr)
            log(f"[Audio] Generated audio for sample {i}: duration={len(gen_audio_np)/gen_sr:.2f}s")
            
            # Log reference audio
            if ref_audio_np is not None and writer:
                writer.add_audio(f"{tag}/reference_audio", normalize_audio(ref_audio_np), global_step=step, sample_rate=sample_rate)
                
            # Generate mel spectrogram figure
            if writer:
                try:
                    mel_gen = compute_mel_spectrogram(gen_audio_np, gen_sr)
                    mel_ref = compute_mel_spectrogram(ref_audio_np, sample_rate) if ref_audio_np is not None else None
                    fig = create_mel_figure(gen_audio_np, mel_gen, gen_sr, step, ref_audio_np, mel_ref)
                    writer.add_figure(f"{tag}/mel_spectrogram", fig, global_step=step)
                    log(f"[Audio] Created mel spectrogram figure for sample {i}")
                except Exception as e:
                    logger.warning(f"[Warning] Failed to create mel spectrogram: {e}")
                
        except Exception as e:
            logger.warning(f"[Warning] Failed to generate audio for sample {i}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                model.audio_vae = None
                audio_vae.to("cpu")
                if prev_training:
                    model.train()
                else:
                    model.eval()
            except Exception as e:
                logger.warning(f"[Warning] Failed to restore model state: {e}")


def save_checkpoint(model, optimizer, scheduler, save_dir: Path, step: int, pretrained_path: str = None, hf_model_id: str = "", distribute: bool = False, accelerator=None):
    """
    Save checkpoint with different strategies for full finetune vs LoRA:
    - Full finetune: save non-vae weights to model.safetensors (or pytorch_model.bin if safetensors unavailable)
    - LoRA: save only lora weights to lora_weights.safetensors (or lora_weights.ckpt if safetensors unavailable)
    """
    import shutil
    tag = "latest_state" if step == 0 else f"step_{step:07d}"
    folder = save_dir / tag
    if accelerator.is_main_process:
        latest_dir = save_dir / "latest_state"
        latest_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        folder.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.save_state(str(folder))

    # Save custom step info
    if accelerator.is_main_process:
        step_info = {"step": step}
        with open(folder / "custom_checkpoint_info.json", "w") as f:
            json.dump(step_info, f)
            
        unwrapped = accelerator.unwrap_model(model)
        lora_cfg = unwrapped.lora_config
        
        if lora_cfg is not None:
            # Save LoRA config and base model path to a separate JSON file
            base_model_to_save = hf_model_id if distribute else (str(pretrained_path) if pretrained_path else None)
            lora_info = {
                "base_model": base_model_to_save,
                "lora_config": lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg),
            }
            with open(folder / "lora_config.json", "w", encoding="utf-8") as f:
                json.dump(lora_info, f, indent=2, ensure_ascii=False)
        else:
            # Copy config files from pretrained path
            if pretrained_path:
                pretrained_dir = Path(pretrained_path)
                files_to_copy =[
                    "config.json", 
                    "audiovae.pth", 
                    "audiovae.safetensors",
                    "tokenizer.json", 
                    "special_tokens_map.json", 
                    "tokenizer_config.json"
                ]
                for fname in files_to_copy:
                    src = pretrained_dir / fname
                    if src.exists():
                        shutil.copy2(src, folder / fname)
    accelerator.wait_for_everyone()


from datasets import Dataset
def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVoxCPMDataset(hf_dataset)

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
    if config_file:
        yaml_args = load_yaml_config(config_file)
        train(**yaml_args)
    else:
        with argbind.scope(args):
            train()