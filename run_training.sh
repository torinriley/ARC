#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <command> [checkpoint_path]"
    echo "Commands:"
    echo "  start   - Start new training"
    echo "  resume  - Resume from checkpoint (requires checkpoint_path)"
    echo "  pull    - Pull latest checkpoints from HF Hub"
    exit 1
fi

# Environment setup
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.huggingface/token)}  # Load from file if not set
export WANDB_API_KEY=${WANDB_API_KEY:-$(cat ~/.wandb/api_key)}  # Load from file if not set
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29500

# Docker configuration
DOCKER_ARGS="--gpus all \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HF_TOKEN=${HF_TOKEN} \
    -e WANDB_API_KEY=${WANDB_API_KEY}"

command=$1
checkpoint_path=$2

case $command in
    "start")
        echo "Starting new training..."
        docker run ${DOCKER_ARGS} arc-training \
            deepspeed --num_gpus=4 train.py
        ;;
    "resume")
        if [ -z "$checkpoint_path" ]; then
            echo "Error: checkpoint_path is required for resume command"
            exit 1
        fi
        echo "Resuming from checkpoint: $checkpoint_path"
        docker run ${DOCKER_ARGS} arc-training \
            deepspeed --num_gpus=4 train.py --resume_from "$checkpoint_path"
        ;;
    "pull")
        echo "Pulling latest checkpoints from HuggingFace Hub..."
        docker run ${DOCKER_ARGS} arc-training \
            python -c "from huggingface_hub import snapshot_download; snapshot_download('torinriley/ARC', repo_type='model')"
        ;;
    *)
        echo "Unknown command: $command"
        exit 1
        ;;
esac
