import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from model import Transformer 
from pathlib import Path
import json

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def save_checkpoint(model, optimizer, epoch, config, accelerator):
    checkpoint_path = Path(config["checkpoint_dir"]) / f"model_epoch_{epoch}.pt"
    accelerator.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved at {checkpoint_path}")

# Preprocess data by tokenizing and padding/truncating to max sequence length
def preprocess_data(example, tokenizer, max_seq_len):
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt")
    return {"input_tokens": tokens["input_ids"].squeeze(0), "attention_mask": tokens["attention_mask"].squeeze(0)}

# Dataset wrapper to convert HuggingFace datasets to PyTorch DataLoader format
class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return {
            "input_tokens": item["input_tokens"],
            "attention_mask": item["attention_mask"],
        }

# Training loop for a single epoch
def train_epoch(model, dataloader, criterion, optimizer, accelerator):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_tokens = batch["input_tokens"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)

        # Forward pass
        outputs = model(input_tokens, attention_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), input_tokens.view(-1))
        accelerator.backward(loss)

        # Backward pass and optimization step
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# Main training function
def train(config_path):
    config = load_config(config_path)
    accelerator = Accelerator(fp16=True)  # Enable mixed precision for efficiency

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    # Stream and preprocess The Pile dataset
    raw_dataset = load_dataset("EleutherAI/pile", streaming=True, split="train")
    processed_dataset = raw_dataset.map(
        lambda x: preprocess_data(x, tokenizer, config["max_seq_len"]),
        batched=True,
        remove_columns=["text"],
    )

    dataset = HuggingFaceDataset(processed_dataset)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize the Transformer model
    model = Transformer.buildTransformer(
        vocab_size=tokenizer.vocab_size,
        seq_len=config["max_seq_len"],
        d_model=config["model_dim"],
        N=config["num_layers"],
        h=config["num_heads"],
        dropout=config["dropout"],
        d_ff=config["feedforward_dim"],
    )
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency
    model, optimizer, dataloader = accelerator.prepare(
        model, optim.AdamW(model.parameters(), lr=config["learning_rate"]), dataloader
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(model, dataloader, criterion, optimizer, accelerator)
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {train_loss:.4f}")

        if (epoch + 1) % config["save_freq"] == 0:
            save_checkpoint(model, optimizer, epoch, config, accelerator)

if __name__ == "__main__":
    config_path = "config.json"
    train(config_path)
