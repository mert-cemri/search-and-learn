#!/bin/bash

hostname --ip-address
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_NAME="Qwen2.5-Math-7B-Instruct"
# MODEL="Qwen/Qwen2.5-Math-72B-Instruct"
# MODEL_NAME="Qwen2.5-Math-72B-Instruct"

if [ -z "$1" ]; then
  echo "No CUDA device specified. Using default: 6"
  CUDA_DEVICE=6
else
  CUDA_DEVICE=$1
fi

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size 1 \
        --port 12341 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-model-len 8192 \
        --enforce-eager \
        --enable_prefix_caching \
        --gpu_memory_utilization 0.95