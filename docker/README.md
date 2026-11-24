# Public build context

This directory is the reference build recipe for the serving image used in the
four-GPU and eight-GPU profiles. It is a **patches-only** package: no model
weights and no 34.8 GB serving image are stored here.

## Base image

The original nightly tag no longer resolves. The immutable digest is public and
is pinned in `build.sh`:

```
vllm/vllm-openai@sha256:b91d90a63b75d00e780287e4236123a281ee3d29122c864a7044a19a680fdf9d
```

## Build chain

1. base digest → `context/config/Dockerfile.moet-speed2` → `kingsing-cli-compat:20261119-literal-image`
2. `kingsing-cli-compat` → `context/scripts/speed9/Dockerfile.bmm_mega` → `kingsing-speed9:bmm-mega-v1`

Run:

```bash
bash docker/build.sh
```

The build needs Docker, a Linux/amd64 builder, and enough disk for the base
image plus the build cache. A GPU is only needed to run the model.

## What is included

- `context/tools/dsv41_sm120/` — Moet/SM120 patch scripts and CUDA helpers
- `context/tools/dsv41_5090/` — optional 5090 build context
- `context/config/Dockerfile.moet-speed2` — stage 1
- `context/scripts/speed9/Dockerfile.bmm_mega` — stage 2
- `context/scripts/serve_moet.py` — reference launcher (copy to your host; set `DS_ROOT`)

## Notes

- Upstream vLLM, DeepGEMM, FlashInfer, and their dependencies remain under their
  own licenses. This package ships the patches and build recipe, not the weights.
- No API keys, SSH keys, model-download tokens, internal IPs, or host paths are
  stored here. Use environment files on your own host.
- A clean-host build is the reproducibility gate: if the build fails on your
  builder, open an issue with the builder OS, Docker version, and build log.
