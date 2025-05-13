from pathlib import Path

def get_config():
    """
    Returns the configuration dictionary for training and inference.
    Loads from config.yaml and provides defaults.
    """
    import yaml
    from pathlib import Path
    
    # Load YAML config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Add additional computed fields
    config.update({
        "d_model": config.get("model_dim", 2048),  # For backward compatibility
        "total_steps": config.get("total_steps", 100000),
        "weight_decay": config.get("weight_decay", 0.1),
    })
    
    return config

def get_weights_path(config, epoch: str):
    """
    Constructs the file path for saving/loading weights.
    """
    weights_dir = config['weights_dir']
    weights_prefix = config['weights_prefix']
    weights_file = f"{weights_prefix}{epoch}.pt"
    return str(Path(weights_dir) / weights_file)
