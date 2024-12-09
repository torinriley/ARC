import os

cache_dir = "/tmp/cache"
os.makedirs(cache_dir, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from model import Transformer
from pathlib import Path
from config import get_config, get_weights_path


def save_checkpoint(model, optimizer, epoch, config, accelerator):
    checkpoint_path = Path(f"model_checkpoint_epoch_{epoch}.pth")

    accelerator.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved at {checkpoint_path}")


def preprocess_data(example, tokenizer, max_seq_len):
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_seq_len,
                       return_tensors="pt")
    return {
        "input_tokens": tokens["input_ids"].squeeze(0),
        "attention_mask": tokens["attention_mask"].squeeze(0),
    }


def train_epoch(model, dataset, tokenizer, criterion, optimizer, accelerator, config):
    model.train()
    total_loss = 0
    batch = []

    for idx, example in enumerate(dataset):
        processed = preprocess_data(example, tokenizer, config["max_seq_len"])
        batch.append(processed)

        if len(batch) == config["batch_size"]:
            optimizer.zero_grad()

            input_tokens = torch.stack([item["input_tokens"] for item in batch]).to(accelerator.device)
            attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(accelerator.device)

            outputs = model(input_tokens, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_tokens.view(-1))
            accelerator.backward(loss)

            optimizer.step()
            total_loss += loss.item()
            batch = []

        if idx >= config.get("max_steps_per_epoch", 10000):
            break

    return total_loss / max(1, (idx + 1) // config["batch_size"])


def train():
    config = get_config()
    accelerator = Accelerator(mixed_precision="fp16")

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_file"])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", streaming=True, split="train", trust_remote_code=True)

    model = Transformer.buildTransformer(
        vocab_size=len(tokenizer),
        seq_len=config["max_seq_len"],
        d_model=config["model_dim"],
        N=config["num_layers"],
        h=config["num_heads"],
        dropout=config["dropout"],
        d_ff=config["feedforward_dim"],
    )
    model.resize_token_embeddings(len(tokenizer))
    model, optimizer = accelerator.prepare(
        model, optim.AdamW(model.parameters(), lr=config["learning_rate"])
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(
            model, raw_dataset, tokenizer, criterion, optimizer, accelerator, config
        )
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {train_loss:.4f}")

        if (epoch + 1) % config["save_freq"] == 0:
            save_checkpoint(model, optimizer, epoch, config, accelerator)


if __name__ == "__main__":
    train()
