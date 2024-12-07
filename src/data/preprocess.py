import os
import json
from pathlib import Path
from tokenizers import Tokenizer

def preprocess_file(file_path, tokenizer, seq_len, output_dir):
    """
    Tokenize and preprocess a single file, then save it.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_data = []
    for line in lines:
        tokens = tokenizer.encode(line.strip()).ids

        # Truncate if too long
        tokens = tokens[:seq_len - 2]

        input_tokens = [tokenizer.token_to_id('[SOS]')] + tokens + [tokenizer.token_to_id('[EOS]')]
        num_padding_tokens = seq_len - len(input_tokens)
        input_tokens += [tokenizer.token_to_id('[PAD]')] * num_padding_tokens

        label_tokens = tokens + [tokenizer.token_to_id('[EOS]')]
        label_tokens += [tokenizer.token_to_id('[PAD]')] * num_padding_tokens

        processed_data.append({
            "input_tokens": input_tokens,
            "label": label_tokens
        })

    output_file = Path(output_dir) / f"{file_path.stem}_processed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f)

def preprocess_dataset(raw_data_dir, tokenizer_file, seq_len, output_dir):
    """
    Preprocess all files in a dataset directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_file)

    for file_name in os.listdir(raw_data_dir):
        file_path = Path(raw_data_dir) / file_name
        if file_path.is_file():
            preprocess_file(file_path, tokenizer, seq_len, output_dir)

if __name__ == "__main__":
    raw_data_dir = "data/raw"
    tokenizer_file = "tokenizer.json"
    seq_len = 128
    output_dir = "data/processed"

    preprocess_dataset(raw_data_dir, tokenizer_file, seq_len, output_dir)
    print("Preprocessing complete!")
    