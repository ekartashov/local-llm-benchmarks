#!/bin/bash
export HF_HOME="/srv/ai/models"
export UVLOOP_NO_IO_URING=1
export VLLM_USE_V1=1 # We need V1 for this model
export VLLM_V1_ENABLED=1

python -m vllm.entrypoints.openai.api_server \
    --model "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 65536 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 4 \
    --enable-sleep-mode \
    --port 30000 > diag_vllm.log 2>&1 &

VLLM_PID=$!
echo "Started vLLM with PID $VLLM_PID"

# Wait for it to initialize a bit
sleep 40

echo "--- FD Audit ---"
echo "Main Process ($VLLM_PID):"
ls -l /proc/$VLLM_PID/fd | grep -i "io_uring" || echo "None"

echo "Child Processes:"
CHILDREN=$(pgrep -P $VLLM_PID)
for child in $CHILDREN; do
    echo "Child $child ($(cat /proc/$child/comm)):"
    ls -l /proc/$child/fd | grep -i "io_uring" || echo "None"
done

# Check if there are any other related processes
echo "All vLLM related processes:"
ps -ef | grep vllm | grep -v grep

kill -9 $VLLM_PID
pkill -9 -P $VLLM_PID
