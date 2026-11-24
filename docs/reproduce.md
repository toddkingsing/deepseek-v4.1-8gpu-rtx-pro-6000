# Reproduce the eight-GPU profile

## 1. Build the serving image

```bash
git clone https://github.com/toddkingsing/deepseek-v4.1-8gpu-rtx-pro-6000.git
cd deepseek-v4.1-8gpu-rtx-pro-6000
bash docker/build.sh
```

The build pins the public vLLM base image by digest and produces
`kingsing-speed9:bmm-mega-v1` locally. No prebuilt image is required.

## 2. Prepare the official weights

Download `deepseek-ai/DeepSeek-V4.1-Flash-1017`, verify the manifest and
SHA-256 files, and mount the model directory read-only at `/model`. Keep the
download cache outside the image.

## 3. Choose the profile

The full launch arguments are in `configs/2026-11-22-eight-gpu-speed9.json`.
The eight-GPU profile uses:

- `--gpus all`
- `--tensor-parallel-size 4`, `--pipeline-parallel-size 2`
- `VLLM_PP_LAYER_PARTITION=20,20`
- `--max-num-seqs 24`
- `--kv-cache-memory-bytes 38654705664`
- `--numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1`
- `VLLM_MOET_PREFILL_PIPE=2`

## 4. Verify

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-flash-1017","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8}'
```

Then verify the image ID, weight manifest hash, GPU device list, KV/slot
configuration, streaming, and a tool call. `systemctl --user active` is not
enough: wait for `/health` and a real completion.

## 5. Measure

Use your own prompt set and output budget. Separate cold-input time to first
token from steady decode. For concurrency, run independent salted 1M inputs and
measure a shared 60-second decode window. Keep the raw JSON.
