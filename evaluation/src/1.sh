#!/bin/bash

# 设置 GPU 使用 (GPU 2 和 4，不跑满单卡)
export CUDA_VISIBLE_DEVICES=2,4

# 启动 vLLM 服务
# --gpu-memory-utilization 0.6: 限制 GPU 显存使用为 60%
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.6 
