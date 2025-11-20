#!/usr/bin/env python3

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import wandb

from basics.templates import TransformerLM, LSTMLM, AdamW, cross_entropy, get_batch, save_checkpoint, load_checkpoint
from basics.templates import gradient_clipping, get_lr_cosine_schedule


def load_dataset_memmap(filepath: str, dtype: np.dtype = np.uint16) -> np.ndarray:
    filepath = Path(filepath)
    
    if filepath.suffix == '.npy':
        dataset = np.load(filepath, mmap_mode='r')
    else:
        file_size = filepath.stat().st_size
        num_elements = file_size // np.dtype(dtype).itemsize
        dataset = np.memmap(filepath, dtype=dtype, mode='r', shape=(num_elements,))
    
    sample_size = min(10000, len(dataset))
    sample = dataset[:sample_size]
    max_val = np.max(sample)
    min_val = np.min(sample)
    
    print(f"Dataset loaded: {len(dataset)} tokens")
    print(f"Sample range: [{min_val}, {max_val}]")
    
    return dataset


def validate_dataset(dataset: np.ndarray, vocab_size: int) -> None:
    sample_size = min(100000, len(dataset))
    sample = dataset[:sample_size]
    max_val = np.max(sample)
    
    if max_val >= vocab_size:
        print(f"WARNING: Found token ID {max_val} >= vocab_size {vocab_size}")
    else:
        print(f"Dataset validation passed: all sampled tokens < vocab_size")


def evaluate_model(
    model,
    val_dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_eval_batches: int = 100,
    model_type: str = 'transformer'
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            try:
                x, y = get_batch(val_dataset, batch_size, context_length, device)
                if model_type == 'lstm':
                    logits, _ = model(x)
                else:
                    logits = model(x)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Warning: Error during validation batch: {e}")
                continue
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def train(
    config: Dict[str, Any],
    train_dataset_path: str,
    val_dataset_path: str,
    checkpoint_dir: str,
    resume_from: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
) -> None:
    model_type = config.get('model_type', 'transformer').lower()
    if model_type not in ['transformer', 'lstm']:
        raise ValueError(f"model_type must be 'transformer' or 'lstm', got '{model_type}'")
    
    vocab_size = config.get('vocab_size', 50257)
    context_length = config.get('context_length', 1024)
    
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    max_iters = config.get('max_iters', 10000)
    eval_interval = config.get('eval_interval', 500)
    save_interval = config.get('save_interval', 1000)
    log_interval = config.get('log_interval', 100)
    num_eval_batches = config.get('num_eval_batches', 100)
    
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    use_lr_schedule = config.get('use_lr_schedule', False)
    max_learning_rate = config.get('max_learning_rate', learning_rate)
    min_learning_rate = config.get('min_learning_rate', learning_rate * 0.1)
    warmup_iters = config.get('warmup_iters', 0)
    cosine_cycle_iters = config.get('cosine_cycle_iters', max_iters)
    
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype_str = config.get('dtype', 'float32')
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    
    dataset_dtype = np.uint16
    
    model_config = config.get('model', {})
    model_kwargs = dict(model_config)
    model_kwargs.update({
        'vocab_size': vocab_size,
        'device': device,
        'dtype': dtype,
    })
    
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Model config: {model_kwargs}")
    print(f"Training: batch_size={batch_size}, lr={learning_rate}, max_iters={max_iters}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("=" * 80)
    
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config
        )
    
    print("\nLoading datasets...")
    train_dataset = load_dataset_memmap(train_dataset_path, dtype=dataset_dtype)
    val_dataset = load_dataset_memmap(val_dataset_path, dtype=dataset_dtype)
    
    validate_dataset(train_dataset, vocab_size)
    validate_dataset(val_dataset, vocab_size)
    
    print("\nInitializing model...")
    if model_type == 'transformer':
        model = TransformerLM(**model_kwargs)
    else:
        model = LSTMLM(**model_kwargs)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    start_iter = 0
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        start_iter = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed at iteration {start_iter}")
    else:
        # xavier initialization
        for param in model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    model.train()

    running_loss = 0.0
    last_log_time = time.time()
    iter_num = start_iter - 1
    
    try:
        for iter_num in range(start_iter, max_iters):
            x, y = get_batch(train_dataset, batch_size, context_length, device)
            
            if model_type == 'lstm':
                logits, _ = model(x)
            else:
                logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            if max_grad_norm > 0:
                grad_norm = gradient_clipping(model.parameters(), max_grad_norm)
            
            if use_lr_schedule:
                current_lr = get_lr_cosine_schedule(
                    iter_num + 1,
                    max_learning_rate,
                    min_learning_rate,
                    warmup_iters,
                    cosine_cycle_iters
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if (iter_num + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - last_log_time
                tokens_per_sec = (log_interval * batch_size * context_length) / elapsed
                max_memory = torch.cuda.max_memory_allocated() / (1024**3)
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iter {iter_num + 1:6d} | Loss: {avg_loss:.4f} | "
                      f"Iter time: {elapsed / log_interval:.2f}s | "
                      f"Tokens/sec: {tokens_per_sec:.0f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Grad norm: {grad_norm:.2e} | "
                      f"Max memory: {max_memory:.2f} GB"
                      , flush=True)
                
                if wandb_project:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/grad_norm': grad_norm,
                        'train/max_memory': max_memory,
                        'train/learning_rate': current_lr,
                        'train/tokens_per_sec': tokens_per_sec,
                        'iter': iter_num + 1,
                    })
                
                running_loss = 0.0
                last_log_time = time.time()
            
            if (iter_num + 1) % eval_interval == 0:
                print(f"\nEvaluating at iteration {iter_num + 1}...")
                val_loss = evaluate_model(
                    model, val_dataset, batch_size, context_length, device, num_eval_batches, model_type
                )
                print(f"Validation loss: {val_loss:.4f}\n")
                
                if wandb_project:
                    wandb.log({
                        'iter': iter_num + 1,
                        'val/loss': val_loss,
                    })
            
            if (iter_num + 1) % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num + 1}.pt"
                print(f"Saving checkpoint to {checkpoint_path}...")
                save_checkpoint(model, optimizer, iter_num + 1, checkpoint_path)
                print("Checkpoint saved.")
        
        print("\nTraining completed!")
        final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
        print(f"Saving final checkpoint to {final_checkpoint_path}...")
        save_checkpoint(model, optimizer, max_iters, final_checkpoint_path)
        print("Final checkpoint saved.")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        interrupt_checkpoint_path = checkpoint_dir / f"checkpoint_interrupt_iter_{iter_num + 1}.pt"
        print(f"Saving interrupt checkpoint to {interrupt_checkpoint_path}...")
        save_checkpoint(model, optimizer, iter_num + 1, interrupt_checkpoint_path)
        print("Interrupt checkpoint saved.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    if wandb_project:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
    )
    
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
    )
    
    parser.add_argument(
        '--val-data',
        type=str,
        required=True,
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
    )
    
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
    )
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.config).parent.parent.parent / "outputs" / "checkpoints" / config['model_type'] / date_time
    train(
        config=config,
        train_dataset_path=args.train_data,
        val_dataset_path=args.val_data,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume_from,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )


if __name__ == '__main__':
    main()
