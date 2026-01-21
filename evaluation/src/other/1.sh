# 只使用 GPU 4
export CUDA_VISIBLE_DEVICES=1,2

# 启动 vLLM OpenAI API Server，端口 8008
python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --host 0.0.0.0 \
    --port 8008 \
