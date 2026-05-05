# vLLM Deployment Procedures

## Critical environment variables (all required, set in deploy.sh or exported)

```bash
VLLM_USE_V1=0                          # Disable V1 engine — stability fix for Blackwell sm_120
VLLM_SERVER_DEV_MODE=1                     # Exposes /sleep, /wake_up, /is_sleeping routes
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 # Required for TP=2 -- prevents OOM during graph capture
UV_USE_IO_URING=0                          # Required if CRIU checkpointing is planned
```

## Port assignments

| Role | Container name | Port | GPU(s) | TP |
|------|---------------|------|--------|-----|
| Arclight Coder | arclight-coder (or bench-vllm-tp2a) | 30000 | GPU0+1 | 2 |
| Arclight Thinker | arclight-thinker | 30001 | GPU1 | 1 |
| Extended Arclight | (same container as coder, restarted) | 30000 | GPU0+1 | 2 |
| Convergence | convergence | 8002 | CPU + GPU (attn) | N/A |

## Standard deploy commands

### Arclight Coder (TP=2, production)
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --gpu-mem-util 0.90 \
  --ctx 32768 \
  --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

### Arclight Thinker (TP=1 GPU1, production)
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
./infra/scripts/deploy.sh vllm gpu1 QuantTrio/Qwen3.6-27B-AWQ \
  --gpu-mem-util 0.90 \
  --ctx 32768 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

### Extended Arclight Coder (TP=2, 65K context)
```bash
VLLM_USE_V1=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 UV_USE_IO_URING=0 \
./infra/scripts/deploy.sh vllm tp2a cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --gpu-mem-util 0.90 \
  --ctx 65536 \
  --kv-cache-dtype fp8 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

## Sleep/wake commands

```bash
# Sleep thinker (frees ~21GB GPU1 VRAM)
curl -s -X POST http://localhost:30001/sleep?level=1
curl -s http://localhost:30001/is_sleeping  # verify: {"is_sleeping": true}

# Wake thinker
curl -s -X POST http://localhost:30001/wake_up

# Sleep coder
curl -s -X POST http://localhost:30000/sleep?level=1
```

## Model downloads (host only — requires pyenv hf)

```bash
pyenv activate hf
HF_HOME=/srv/ai/models hf download QuantTrio/Qwen3.6-27B-AWQ
HF_HOME=/srv/ai/models hf download cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
# Gated models:
HF_TOKEN=hf_... HF_HOME=/srv/ai/models hf download mistralai/Devstral-Small-2505
```

Do not use throwaway Python containers for downloads — the host pyenv has everything.

## Podman compose backend

All engines run as rootless podman containers. Scripts use `podman compose`, not `docker compose`. GPU isolation uses `NVIDIA_VISIBLE_DEVICES` (not `CUDA_VISIBLE_DEVICES`). See `infra/compose/` for templates; `infra/scripts/deploy.sh` is the canonical entry point.

## Common failure modes

### OOM during CUDA graph capture
- Add `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`
- If still failing, reduce `--gpu-memory-utilization` (try 0.85 → 0.80)

### OOM on startup with NVFP4 MoE model (PrismaQuant 35B A3B)
The OOM is in the **profiling forward pass**, not CUDA graph capture. `gpu-mem-util` has no effect because model weights alone (~29.4 GiB) already exceed any budget. The pass size is controlled by `max_num_seqs`.
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./infra/scripts/deploy.sh vllm gpu0 rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm \
  --trust-remote-code \
  --gpu-mem-util 0.90 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 16 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```
Notes:
- `--gpu-mem-util 0.90` required (0.84 is below the model floor → 0 KV blocks).
- CUDA graphs work with this config: `enforce_eager=False`, `CUDAGraphMode.FULL_AND_PIECEWISE`. TPS: 120.9 t/s decode (N=1) / 459.3 t/s agg (N=4). BENCH_23 2026-05-05.

### TP=1 gives ~20 t/s instead of ~230 t/s
- Check if V1 engine is active (`VLLM_USE_V1=0` may not be set)
- V1 forces eager mode on Blackwell in some configs → 10× penalty

### Tool calls returning no_call or format_error
- Verify `--tool-call-parser` is set correctly (see docs/decisions/models.md parser table)
- Verify `--enable-auto-tool-choice` is present
- For thinking models, verify `--reasoning-parser` is correct (or absent for hermes/gemma4 models)
- Check `delta.reasoning` vs `delta.reasoning_content` in bench client if reasoning not captured

### Container won't start with --enable-sleep-mode
- Both are required: env `VLLM_SERVER_DEV_MODE=1` AND serve flag `--enable-sleep-mode`
- Without the serve flag, `/sleep` is a no-op (VRAM never freed)

### VRAM not freed after sleep
- Confirm `--enable-sleep-mode` is in the vllm serve command (not just the env var)
- Level=1 frees ~92%. Level=2 is broken (gibberish on wake — do not use)
- Level=1 retains ~4 GiB residual (CUDA graphs) — this is normal and cannot be reduced
