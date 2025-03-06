#!/bin/bash

hostname --ip-address
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
MODEL="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
MODEL_NAME="Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
# MODEL="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
# MODEL_NAME="Skywork-o1-Open-PRM-Qwen-2.5-7B"

if [ -z "$1" ]; then
  echo "No CUDA device specified. Using default: 5"
  CUDA_DEVICE=5
else
  CUDA_DEVICE=$1
fi

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size 1 \
        --port 12342 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-model-len 8192 \
        --enforce-eager \
        --enable_prefix_caching \
        --gpu_memory_utilization 0.95