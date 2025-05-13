import os
import time
import json
import wandb
import torch
import torch.nn.functional as F
import deepspeed
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from huggingface_hub import HfApi
from datetime import datetime
from datasets import load_dataset, interleave_datasets, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from models.mixtral import MixtralModel
from accelerate import Accelerator
from config import get_config

# Load DeepSpeed config
with open('ds_config.json', 'r') as f:
    deepspeed_config = json.load(f)

CHECKPOINT_DIR = "checkpoints"
HF_REPO_ID = "torinriley/ARC"
SAVE_INTERVAL = 1800

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TrainingState:
    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.last_save_time = time.time()
        self.train_losses = AverageMeter()
        self.aux_losses = AverageMeter()
        self.val_losses = AverageMeter()  
        self.expert_metrics = {}
        
    def update_step(self, prediction_loss, aux_loss, expert_stats=None):
        self.step += 1
        self.train_losses.update(prediction_loss)
        self.aux_losses.update(aux_loss)
        if expert_stats:
            for k, v in expert_stats.items():
                if k not in self.expert_metrics:
                    self.expert_metrics[k] = AverageMeter()
                self.expert_metrics[k].update(v)
        
        # Monitor expert utilization
        if expert_stats and 'expert_counts' in expert_stats:
            counts = expert_stats['expert_counts']
            total = counts.sum().item()
            utilization = (counts > 0).float().mean().item()
            if utilization < 0.8:  # Alert if less than 80% experts used
                print(f"Warning: Expert utilization at {utilization:.2%}")

def setup_tokenizer(vocab_size: int = 256_000) -> PreTrainedTokenizerFast:
    """Initialize or load tokenizer with safety checks"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, trust_remote_code=True)
        assert tokenizer.vocab_size >= vocab_size, "Tokenizer too small"
    except Exception as e:
        print(f"Loading default tokenizer due to: {str(e)}")
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1",
            model_max_length=2048,
            trust_remote_code=True
        )
    return tokenizer

def setup_wandb(config: Dict[str, Any]):
    """Enhanced W&B logging with system monitoring"""
    wandb.init(
        project="arc-training",
        config=config,
        name=f"mixtral-{config['d_model']}-{config['n_layers']}l",
        settings=wandb.Settings(console="wrap")
    )
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")

def create_dataloaders(tokenizer, batch_size: int, max_length: int):
    """Improved dataset pipeline with length filtering and proper interleaving"""
    datasets = {
        "anthropic": load_dataset("anthropic/hh-rlhf", streaming=True, split="train").take(10_000),
        "stack": load_dataset("bigcode/starcoderdata", streaming=True, split="train").take(50_000),
        "redpajama": load_dataset("togethercomputer/RedPajama-Data-1T-Sample", streaming=True, split="train").take(100_000),
        "books": load_dataset("emozilla/booksum", streaming=True, split="train").repeat(),
        "wiki": load_dataset("wikipedia", "20220301.en", streaming=True, split="train").repeat(),
    }
    
    # Prefetch buffer for streaming efficiency
    for name in datasets:
        datasets[name] = datasets[name].map(
            lambda x: x,
            remove_columns=datasets[name].column_names,
            batched=True,
            batch_size=1000,
            num_proc=4
        )
    
    def tokenize_and_filter(example):
        """Tokenize with length filtering"""
        text = extract_text(example)
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=True,
            return_tensors="pt"
        )
        
        # Filter short sequences
        valid_indices = [i for i, ids in enumerate(tokenized["input_ids"]) 
                        if len(ids) >= 256]
        
        return {
            "input_ids": [tokenized["input_ids"][i] for i in valid_indices],
            "attention_mask": [tokenized["attention_mask"][i] for i in valid_indices]
        }
    
    # Process and interleave
    processed_datasets = [
        datasets[name].map(tokenize_and_filter, batched=True, remove_columns=datasets[name].column_names)
        for name in datasets
    ]
    
    combined_dataset = interleave_datasets(
        processed_datasets,
        probabilities=[0.3, 0.2, 0.2, 0.2, 0.1],
        seed=42,
        stopping_strategy="all_exhausted"
    )
    
    return torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

def extract_text(example: Dict) -> str:
    """Robust text extraction from different dataset formats"""
    if "text" in example:
        return example["text"].strip()
    elif "content" in example:
        return example["content"].strip()
    elif all(k in example for k in ["instruction", "output"]):
        return f"Instruction: {example['instruction']}\nOutput: {example['output']}".strip()
    return str(example).strip()

def validate(model: deepspeed.DeepSpeedEngine, val_loader: torch.utils.data.DataLoader) -> float:
    """Run validation loop"""
    model.eval()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(model.device)
            outputs = model(input_ids=input_ids)
            loss = F.cross_entropy(
                outputs["logits"][..., :-1, :].contiguous().view(-1, outputs["logits"].size(-1)),
                input_ids[..., 1:].contiguous().view(-1)
            )
            loss_meter.update(loss.item())
    
    model.train()
    return loss_meter.avg

def load_checkpoint(
    model: deepspeed.DeepSpeedEngine,
    tokenizer: PreTrainedTokenizerFast,
    checkpoint_path: str,
    training_state: TrainingState
) -> Tuple[deepspeed.DeepSpeedEngine, PreTrainedTokenizerFast, TrainingState]:
    """Load model, tokenizer and training state from checkpoint"""
    try:
        # Load tokenizer first as it's simpler
        if os.path.exists(os.path.join(checkpoint_path, "tokenizer.json")):
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        # Load DeepSpeed checkpoint
        _, client_state = model.load_checkpoint(checkpoint_path)
        
        # Restore training state if available
        if client_state:
            training_state.step = client_state.get('step', 0)
            training_state.epoch = client_state.get('epoch', 0)
            training_state.best_loss = client_state.get('best_loss', float('inf'))
            training_state.last_save_time = time.time()
            print(f"Resumed from step {training_state.step} with best loss {training_state.best_loss}")
        
        return model, tokenizer, training_state
    except Exception as e:
        raise Exception(f"Failed to load checkpoint: {str(e)}")

def save_checkpoint(
    model: deepspeed.DeepSpeedEngine,
    tokenizer: PreTrainedTokenizerFast,
    training_state: TrainingState,
    is_best: bool = False
):
    """Enhanced checkpointing with validation tracking and state saving"""
    try:
        # Prepare checkpoint path
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"step_{training_state.step}")
        
        # Save training state
        client_state = {
            'step': training_state.step,
            'epoch': training_state.epoch,
            'best_loss': training_state.best_loss,
            'train_losses': training_state.train_losses.avg,
            'aux_losses': training_state.aux_losses.avg,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save DeepSpeed model with training state
        model.save_checkpoint(checkpoint_path, client_state=client_state)
        
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save metadata
        metadata = {
            'step': training_state.step,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best,
            'best_loss': training_state.best_loss
        }
        with open(os.path.join(checkpoint_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if is_best:
            best_path = os.path.join(CHECKPOINT_DIR, "best")
            model.save_checkpoint(best_path, client_state=client_state)
            tokenizer.save_pretrained(best_path)
            
        # Upload to HF Hub
        api = HfApi()
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        
        print(f"Saved checkpoint at step {training_state.step}")
    except Exception as e:
        print(f"WARNING: Failed to save checkpoint: {str(e)}")

def setup_model(config: Dict[str, Any], tokenizer: PreTrainedTokenizerFast) -> MixtralModel:
    """Initialize model with proper tokenizer compatibility."""
    # Ensure vocab size matches tokenizer
    config["vocab_size"] = len(tokenizer)
    
    # Initialize model
    model = MixtralModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["num_layers"],
        n_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        num_experts=config["num_experts"],
        num_experts_per_tok=config["num_experts_per_tok"]
    )
    
    # Tie weights
    model.lm_head.weight = model.token_emb.weight
    
    # Special token handling
    if hasattr(tokenizer, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(tokenizer, "eos_token_id"):
        model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(tokenizer, "bos_token_id"):
        model.config.bos_token_id = tokenizer.bos_token_id
        
    return model

def load_mixtral_weights(model: MixtralModel, checkpoint_path: str) -> None:
    """Load weights from Mixtral-8x7B checkpoint with compatibility handling."""
    try:
        state_dict = torch.load(checkpoint_path)
        model_state_dict = model.state_dict()
        
        # Handle key mapping
        key_mapping = {
            # Add mappings for different key names if needed
            "model.embed_tokens.weight": "token_emb.weight",
            "model.norm.weight": "norm.weight"
        }
        
        # Load weights with compatibility handling
        missing_keys = []
        unexpected_keys = []
        
        for checkpoint_key, param in state_dict.items():
            model_key = key_mapping.get(checkpoint_key, checkpoint_key)
            
            if model_key in model_state_dict:
                if param.shape == model_state_dict[model_key].shape:
                    model_state_dict[model_key].copy_(param)
                else:
                    print(f"Shape mismatch for {model_key}: expected {model_state_dict[model_key].shape}, got {param.shape}")
                    missing_keys.append(model_key)
            else:
                unexpected_keys.append(checkpoint_key)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
        # Update tied weights
        model.lm_head.weight = model.token_emb.weight
        
    except Exception as e:
        raise Exception(f"Failed to load Mixtral weights: {str(e)}")

def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup model and tokenizer with Mixtral-8x7B compatibility."""
    # First load tokenizer
    tokenizer = setup_tokenizer(config["vocab_size"])
    
    # Initialize model with correct config
    model = setup_model(config, tokenizer)
    
    # Load pretrained weights if specified
    if config.get("pretrained_path"):
        load_mixtral_weights(model, config["pretrained_path"])
        
    return model, tokenizer

def train(resume_from: Optional[str] = None):
    config = get_config()
    config.update({
        "gradient_accumulation_steps": 32,  # Matched with DeepSpeed config
        "max_grad_norm": 1.0,
        "warmup_steps": 4000,  # Matched with optimized config
        "eval_interval": 2000,
    })
    
    # Create checkpoint directory
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    
    # Try to pull latest checkpoint from HF Hub if resuming
    if resume_from:
        try:
            api = HfApi()
            api.snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="model",
                local_dir=CHECKPOINT_DIR
            )
            print(f"Successfully pulled checkpoints from {HF_REPO_ID}")
        except Exception as e:
            print(f"Warning: Failed to pull checkpoints: {str(e)}")
    
    accelerator = Accelerator(mixed_precision="bf16")  # Changed to bf16 to match DeepSpeed config
    setup_wandb(config)

    # Initialize components
    tokenizer = setup_tokenizer(config["vocab_size"])
    model = setup_model(config, tokenizer)
    
    # Training state
    training_state = TrainingState()
    
    # Resume from checkpoint if specified
    if resume_from:
        resume_path = os.path.join(CHECKPOINT_DIR, resume_from)
        if os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            model, tokenizer, training_state = load_checkpoint(
                model, tokenizer, resume_path, training_state
            )
        else:
            print(f"Warning: Checkpoint {resume_path} not found, starting from scratch")
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"]
    )
    
    # DeepSpeed initialization
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=deepspeed_config,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, step / config["warmup_steps"])
        )
    )

    # Data loaders
    train_loader = create_dataloaders(tokenizer, config["batch_size"], config["max_seq_len"])
    val_loader = create_dataloaders(tokenizer, config["batch_size"], config["max_seq_len"])  # NEW: Validation set
    
    # Training state
    training_state = TrainingState()
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    global_step = 0

    # Training loop
    while global_step < config["total_steps"]:
        for batch in train_loader:
            if global_step >= config["total_steps"]:
                break
                
            # Forward pass
            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            
            # Create causal mask
            seq_len = input_ids.size(1)
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device),
                diagonal=1
            )
            
            # Combine with attention mask
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
                attention_mask = attention_mask & ~causal_mask
            
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Loss calculation (with proper masking)
            logits = outputs["logits"][..., :-1, :].contiguous()
            labels = input_ids[..., 1:].contiguous()
            mask = attention_mask[..., 1:].contiguous() if attention_mask is not None else None
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='none'
            ).view_as(labels)
            
            if mask is not None:
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.mean()
            total_loss = loss + outputs["aux_loss"]
            
            # Backward pass with gradient accumulation
            model_engine.backward(total_loss)
            
            if (global_step + 1) % config["gradient_accumulation_steps"] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                model_engine.step()
                model_engine.zero_grad()
                
                # LR scheduling
                lr_scheduler.step()
                
                # Validation
                if global_step % config["eval_interval"] == 0:
                    val_loss = validate(model_engine, val_loader)
                    training_state.val_losses.update(val_loss)
                    is_best = val_loss < training_state.best_loss
                    training_state.best_loss = min(val_loss, training_state.best_loss)
                    
                    wandb.log({
                        "val/loss": val_loss,
                        "val/best_loss": training_state.best_loss,
                    }, step=global_step)
            
            # Logging
            training_state.update_step(loss.item(), outputs["aux_loss"].item())
            if global_step % config["log_interval"] == 0:
                wandb.log({
                    "train/loss": training_state.train_losses.avg,
                    "train/aux_loss": training_state.aux_losses.avg,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": get_grad_norm(model),  # Implement this helper
                    "system/gpu_mem": torch.cuda.max_memory_allocated() / 1e9,
                }, step=global_step)
                training_state.reset_metrics()
            
            # Checkpointing
            if time.time() - training_state.last_save_time >= SAVE_INTERVAL:
                save_checkpoint(model_engine, tokenizer, training_state, is_best)
                training_state.last_save_time = time.time()
            
            global_step += 1

    # Final save
    save_checkpoint(model_engine, tokenizer, training_state, is_best=False)

def get_grad_norm(model):
    """Compute gradient norm for monitoring"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Mixtral MoE model')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    train(resume_from=args.resume_from)