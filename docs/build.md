# Building the image

**You almost certainly don't need to build anything.** The image is prebuilt and
**anonymously pullable** — this is the canonical, supported artifact for every model in
this repo:

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest
```

It's vLLM v0.23.0 compiled from source for GB10 / sm_121a with the full AEON
speculative-decoding stack (Triton NVFP4 KV cache, DFlash SWA / high-concurrency /
prefix-cache fixes, the DGX Spark runtime patches, TurboQuant). See the
[Quickstart](../README.md) and [DGX Spark setup](dgx-spark-setup.md) to deploy it, and
the container repo's
[**Build provenance**](https://github.com/AEON-7/vllm-ultimate-dgx-spark#build-provenance)
for the exact source pins behind each tag (`:latest` = `:2026-06-18-v0.23.0-dflashfix`;
rollback `:2026-06-11-pr41703`).

## Reproducing the image from source (advanced)

The canonical Dockerfile, patches, source pin, verify script, and build provenance all
live in **[AEON-7/vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark)**.
To rebuild it yourself, follow that repo's
[**`SOURCE.md`**](https://github.com/AEON-7/vllm-ultimate-dgx-spark/blob/main/SOURCE.md) —
it has the source pin, the `docker build` step, the build knobs (`MAX_JOBS`,
`TORCH_CUDA_ARCH_LIST=12.1a`, `ENABLE_NVFP4_SM100=0`, …), and build troubleshooting.

> **Note:** an earlier version of this page documented building the old
> `vllm-spark-omni-q36:v1` image (vLLM 0.19.1, sm_120, a text-only `Qwen3_5MoeForCausalLM`
> registry patch). That image and its build are **superseded** by the unified
> `aeon-vllm-ultimate` source-build in the container repo above — which needs no registry
> patch (the multimodal classes load natively).
