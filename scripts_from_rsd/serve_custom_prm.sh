#!/bin/bash

hostname --ip-address
MODEL="/data/user_data/mert/spec/models_merged/Qwen2--qwen_7b_merged-qwen2model"
MODEL_NAME="Qwen2--qwen_7b_merged-qwen2model"

export CUDA_VISIBLE_DEVICES=7
python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size 1 \
        --port 12343 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --enforce-eager \
        --enable_prefix_caching \
        --gpu_memory_utilization 0.5