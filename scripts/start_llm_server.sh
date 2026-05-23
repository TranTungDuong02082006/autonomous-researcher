#!/bin/bash
# Script to launch local vLLM server with Qwen 2.5

MODEL_NAME=${1:-"Qwen/Qwen2.5-14B-Instruct"}
PORT=${2:-8000}
QUANTIZATION=${3:-"gptq"} # or awq, or None

echo "=== Starting vLLM Server ==="
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Quantization: $QUANTIZATION"

vllm serve "$MODEL_NAME" \
  --port "$PORT" \
  --quantization "$QUANTIZATION" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
