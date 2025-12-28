# -*- coding: utf-8 -*-
"""
Progressive LoRA Merging (PLM)
Complete model identity replacement via iterative train-merge cycles.

Paper: "Body Snatching: Complete Model Identity Replacement via Progressive LoRA Merging"
Author: Ouissam Said Drissi (wissam.idrissi@gmail.com)

Usage:
    python plm.py --base-model Qwen/Qwen3-1.7B --dataset your_data.jsonl --cycles 100
    python plm.py --base-model meta-llama/Llama-3-8B --dataset data.jsonl --cycles 50

The key insight: Catastrophic forgetting is a FEATURE, not a bug.
Each cycle permanently merges learned weights into the base, progressively
replacing the model's original identity with your data.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datasets import Dataset
import json
import pandas as pd
from tqdm import tqdm
import random
import shutil
from pathlib import Path
import gc
import argparse
import os
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "lora_r": 8,                    # LoRA rank (small is fine, we accumulate over cycles)
    "lora_alpha": 32,               # LoRA alpha (4:1 ratio with rank)
    "lora_dropout": 0.05,           # Light dropout
    "learning_rate": 1e-4,          # Standard LoRA learning rate
    "epochs_per_cycle": 1,          # Epochs before each merge
    "batch_size": 1,                # Per-device batch size
    "gradient_accumulation": 4,     # Effective batch = batch_size * this
    "max_length": 4096,             # Max sequence length
    "warmup_steps": 50,             # Warmup steps per cycle
    "save_every_n_cycles": 5,       # Save checkpoint every N cycles
    "output_dir": "./plm_output",   # Output directory
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_jsonl(file_path: str, tokenizer, max_length: int = 4096) -> List[str]:
    """
    Load dataset from JSONL file.
    
    Expected format (any of these):
        {"text": "full conversation text"}
        {"prompt": "...", "response": "..."}
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    print(f"\nLoading dataset from {file_path}...")
    
    texts = []
    skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [Skip] Line {line_num}: Invalid JSON - {str(e)[:50]}")
                skipped += 1
                continue
            
            # Handle different formats
            if 'text' in data:
                text = data['text']
            elif 'training_data' in data:
                text = data['training_data']
            elif 'prompt' in data and 'response' in data:
                # Convert to chat format
                text = f"<|im_start|>user\n{data['prompt']}<|im_end|>\n<|im_start|>assistant\n{data['response']}<|im_end|>"
            elif 'messages' in data:
                # Convert messages array to text
                text = ""
                for msg in data['messages']:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                text = text.strip()
            else:
                print(f"  [Skip] Line {line_num}: Unknown format - {list(data.keys())}")
                skipped += 1
                continue
            
            # Check length
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            if token_count > max_length:
                skipped += 1
                continue
            
            texts.append(text)
    
    print(f"  Loaded: {len(texts)} examples")
    if skipped > 0:
        print(f"  Skipped: {skipped} examples")
    
    random.shuffle(texts)
    return texts


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_4bit(model_path: str):
    """Load model in 4-bit quantization for memory-efficient training."""
    
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    print(f"\n=== Loading Model (4-bit) ===")
    print(f"Model: {model_path}")
    print(f"Compute dtype: {'BF16' if use_bf16 else 'FP16'}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"  Loaded successfully")
    print(f"  Vocab size: {len(tokenizer)}")
    
    return model, tokenizer


def load_model_full_precision(model_path: str, tokenizer):
    """Load model in full precision (BF16) for merging."""
    
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    print(f"\n=== Loading Model (Full Precision for Merge) ===")
    print(f"Model: {model_path}")
    print(f"Dtype: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cpu",  # CPU for merge to save VRAM
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Resize embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    return model


# =============================================================================
# LORA SETUP
# =============================================================================

def apply_lora(model, config: dict):
    """Apply fresh LoRA adapters to model."""
    
    print(f"\n=== Applying LoRA ===")
    print(f"  Rank: {config['lora_r']}, Alpha: {config['lora_alpha']}")
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model


# =============================================================================
# MERGING
# =============================================================================

def merge_lora_high_precision(adapter_path: str, base_model_path: str, output_path: str, tokenizer):
    """
    Merge LoRA adapter into base model using high precision (BF16).
    
    CRITICAL: Always merge in full precision, never in 4-bit!
    """
    print(f"\n=== Merging LoRA (High Precision) ===")
    print(f"  Base: {base_model_path}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output: {output_path}")
    
    # Load base in full precision
    base_model = load_model_full_precision(base_model_path, tokenizer)
    
    # Apply adapter
    print("  Applying adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge
    print("  Merging weights...")
    merged = model.merge_and_unload()
    
    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    print(f"  Saved to: {output_dir}")
    
    # Cleanup
    del merged, model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return str(output_dir)


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize_for_training(examples: dict, tokenizer, max_length: int) -> dict:
    """Tokenize with causal LM labels."""
    
    encodings = tokenizer(
        examples["text"],
        max_length=max_length,
        padding=False,
        truncation=True,
        return_tensors=None,
    )
    
    # For causal LM, labels = input_ids
    encodings["labels"] = encodings["input_ids"].copy()
    
    return encodings


@dataclass
class DataCollator:
    """Collator that handles padding."""
    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# =============================================================================
# TRAINING
# =============================================================================

class ProgressCallback(TrainerCallback):
    """Simple progress tracking."""
    
    def __init__(self, cycle: int):
        self.cycle = cycle
        self.losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.losses.append(logs['loss'])
            avg = sum(self.losses[-50:]) / min(50, len(self.losses))
            print(f"\r  [Cycle {self.cycle}] Step {state.global_step} | Loss: {logs['loss']:.4f} | Avg: {avg:.4f}", end="")


def train_one_cycle(model, tokenizer, texts: List[str], cycle: int, config: dict):
    """Train for one cycle (one or more epochs)."""
    
    print(f"\n{'='*60}")
    print(f"CYCLE {cycle}")
    print(f"{'='*60}")
    print(f"  Examples: {len(texts)}")
    
    # Create dataset
    df = pd.DataFrame({"text": texts})
    train_size = int(0.95 * len(df))
    
    train_dataset = Dataset.from_pandas(df[:train_size])
    eval_dataset = Dataset.from_pandas(df[train_size:])
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_for_training(x, tokenizer, config['max_length']),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_for_training(x, tokenizer, config['max_length']),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    
    # Training args
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    training_args = TrainingArguments(
        output_dir=f"{config['output_dir']}/cycle_{cycle}",
        num_train_epochs=config['epochs_per_cycle'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation'],
        warmup_steps=config['warmup_steps'],
        learning_rate=config['learning_rate'],
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        gradient_checkpointing=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollator(tokenizer),
        callbacks=[ProgressCallback(cycle)],
    )
    
    # Train
    trainer.train()
    print()  # Newline after progress
    
    # Get final loss
    eval_results = trainer.evaluate()
    print(f"  Eval Loss: {eval_results['eval_loss']:.4f}")
    
    return model, eval_results['eval_loss']


# =============================================================================
# MAIN PROGRESSIVE LOOP
# =============================================================================

def progressive_lora_merge(
    base_model: str,
    dataset_path: str,
    num_cycles: int,
    config: dict = None
) -> str:
    """
    Main Progressive LoRA Merging loop.
    
    For each cycle:
    1. Load base model (4-bit for training)
    2. Apply fresh LoRA
    3. Train
    4. Save adapter
    5. Merge in high precision (BF16)
    6. Use merged as new base
    7. Repeat
    
    Returns path to final merged model.
    """
    
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("PROGRESSIVE LORA MERGING")
    print("="*60)
    print(f"Base Model: {base_model}")
    print(f"Dataset: {dataset_path}")
    print(f"Cycles: {num_cycles}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # Track state
    current_base = base_model
    best_loss = float('inf')
    best_cycle = 0
    
    # Initial model load to get tokenizer
    model, tokenizer = load_model_4bit(base_model)
    
    # Load dataset
    texts = load_dataset_jsonl(dataset_path, tokenizer, config['max_length'])
    if len(texts) == 0:
        raise ValueError("No valid examples in dataset!")
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump({
            "base_model": base_model,
            "dataset": dataset_path,
            "num_cycles": num_cycles,
            "config": config,
            "started": datetime.now().isoformat()
        }, f, indent=2)
    
    # Main loop
    for cycle in range(1, num_cycles + 1):
        
        # Apply fresh LoRA
        if cycle == 1:
            model = apply_lora(model, config)
        else:
            # Reload from merged base
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            model, tokenizer = load_model_4bit(current_base)
            model = apply_lora(model, config)
        
        # Train
        random.shuffle(texts)  # Reshuffle each cycle
        model, eval_loss = train_one_cycle(model, tokenizer, texts, cycle, config)
        
        # Track best
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_cycle = cycle
            print(f"  ★ New best loss!")
        
        # Save adapter
        adapter_path = output_dir / f"adapters/cycle_{cycle}"
        adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        
        # Merge
        merged_path = output_dir / f"merged/cycle_{cycle}"
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        merge_lora_high_precision(
            str(adapter_path),
            current_base,
            str(merged_path),
            tokenizer
        )
        
        # Update base for next cycle
        current_base = str(merged_path)
        
        # Periodic checkpoint
        if cycle % config['save_every_n_cycles'] == 0:
            checkpoint_path = output_dir / "checkpoints" / f"cycle_{cycle}"
            shutil.copytree(merged_path, checkpoint_path, dirs_exist_ok=True)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old merged (keep disk space manageable)
        if cycle > 1:
            old_merged = output_dir / f"merged/cycle_{cycle-1}"
            if old_merged.exists() and cycle % config['save_every_n_cycles'] != 1:
                shutil.rmtree(old_merged)
        
        print(f"  Cycle {cycle} complete. New base: {current_base}")
    
    # Final save
    final_path = output_dir / "final_model"
    shutil.copytree(current_base, final_path, dirs_exist_ok=True)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total cycles: {num_cycles}")
    print(f"Best loss: {best_loss:.4f} (cycle {best_cycle})")
    print(f"Final model: {final_path}")
    print("="*60)
    
    # Save final state
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            "total_cycles": num_cycles,
            "best_loss": best_loss,
            "best_cycle": best_cycle,
            "final_model": str(final_path),
            "completed": datetime.now().isoformat()
        }, f, indent=2)
    
    return str(final_path)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Progressive LoRA Merging - Complete model identity replacement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plm.py --base-model Qwen/Qwen3-1.7B --dataset data.jsonl --cycles 100
  python plm.py --base-model meta-llama/Llama-3-8B --dataset data.jsonl --cycles 50 --lora-r 16

Dataset format (JSONL, any of these):
  {"text": "full conversation text"}
  {"prompt": "user input", "response": "assistant output"}
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Paper: "Body Snatching: Complete Model Identity Replacement via Progressive LoRA Merging"
Author: Ouissam Said Drissi (wissam.idrissi@gmail.com)
        """
    )
    
    # Required
    parser.add_argument("--base-model", required=True, help="Base model path or HF model ID")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--cycles", type=int, required=True, help="Number of train-merge cycles")
    
    # Optional
    parser.add_argument("--output-dir", default="./plm_output", help="Output directory")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--epochs-per-cycle", type=int, default=1, help="Epochs per cycle")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N cycles")
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config.update({
        "output_dir": args.output_dir,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "epochs_per_cycle": args.epochs_per_cycle,
        "save_every_n_cycles": args.save_every,
    })
    
    # Run
    final_model = progressive_lora_merge(
        base_model=args.base_model,
        dataset_path=args.dataset,
        num_cycles=args.cycles,
        config=config
    )
    
    print(f"\nDone! Final model at: {final_model}")


if __name__ == "__main__":
    main()
