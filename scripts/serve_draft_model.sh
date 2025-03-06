#!/bin/bash

hostname --ip-address
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME="Qwen2.5-Math-1.5B-Instruct"
# MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
# MODEL_NAME="Qwen2.5-Math-7B-Instruct"

# Check if a CUDA device is provided as an argument
if [ -z "$1" ]; then
  echo "No CUDA device specified. Using default: 4"
  CUDA_DEVICE=4
else
  CUDA_DEVICE=$1
fi

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size 1 \
        --port 12340 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-model-len 8192 \
        --enforce-eager \
        --enable_prefix_caching \
        --gpu_memory_utilization 0.95
        