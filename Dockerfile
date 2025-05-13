FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create cache directory for HF
RUN mkdir -p /root/.cache/huggingface

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV WANDB_PROJECT=arc_training
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Run script with DeepSpeed configuration
CMD ["deepspeed", "--num_gpus=4", "train.py"]
