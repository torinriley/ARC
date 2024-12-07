from pathlib import Path

def get_config():
    """
    Returns the configuration dictionary for training and inference.
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "learning_rate": 1e-4,
        "max_seq_len": 350,
        "model_dim": 512,
        "weights_dir": "weights",
        "weights_prefix": "llm_model_",
        "preload_weights": None,
        "tokenizer_file": "tokenizer.json",
        "experiment_dir": "runs/llm"
    }

def get_weights_path(config, epoch: str):
    """
    Constructs the file path for saving/loading weights.
    """
    weights_dir = config['weights_dir']
    weights_prefix = config['weights_prefix']
    weights_file = f"{weights_prefix}{epoch}.pt"
    return str(Path(weights_dir) / weights_file)
