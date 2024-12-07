import torch
from tokenizers import Tokenizer
from pathlib import Path

def load_model(model_path, device):
    """
    Load a trained model from disk.
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def load_tokenizer(tokenizer_file):
    """
    Load the tokenizer.
    """
    tokenizer = Tokenizer.from_file(tokenizer_file)
    return tokenizer

def beam_search_decode(model, tokenizer, max_len, beam_width, device):
    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    sequences = [[torch.tensor([sos_id], device=device), 0]]  # [sequence, score]

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1].item() == eos_id:
                all_candidates.append((seq, score))
                continue
            mask = torch.triu(torch.ones((seq.size(0), seq.size(0)), device=device)) == 1
            output = model(seq.unsqueeze(0), mask.unsqueeze(0))
            logits = output[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, beam_width)

            for i in range(beam_width):
                candidate = torch.cat([seq, torch.tensor([topk.indices[0, i].item()], device=device)])
                candidate_score = score - torch.log(topk.values[0, i]).item()
                all_candidates.append((candidate, candidate_score))

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]  # Keep best beams

    return sequences[0][0]

def generate_text(prompt, model, tokenizer, device):
    """
    Generate text from a prompt using the trained model.
    """
    input_tokens = tokenizer.encode(prompt).ids
    generated_tokens = beam_search_decode(model, input_tokens, tokenizer, max_len=128, beam_width=5, device=device)
    return tokenizer.decode(generated_tokens.tolist(), skip_special_tokens=True)

# Get configuration
def get_config():
    return {
        "seq_len": 128,
        "tokenizer_file": "tokenizer.json",
        "model_path": "llm_model.pt" 
    }

if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config["model_path"], device)
    tokenizer = load_tokenizer(config["tokenizer_file"])

    input_prompt = "Once upon a time"
    generated_text = generate_text(input_prompt, model, tokenizer, config, device)

    print(f"Input: {input_prompt}")
    print(f"Generated: {generated_text}")
