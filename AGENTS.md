# AGENTS.md — Operator's Manual for AI Agents

This file is for AI coding/ops agents deploying **`AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`** with **DFlash** speculative decoding on **NVIDIA DGX Spark (GB10 / sm_121a)**. The settings here are measured and deliberate. Before "improving" a flag, read the matching **DO NOT UNDO** entry — most obvious-looking optimizations have already been tried and regress or break this stack.

---

## ⚠️ Hardware scope

This is tuned **only** for the DGX Spark (GB10, 128 GB unified LPDDR5X, sm_121a Blackwell). The image is source-built for `sm_121a`; it will not run as-is on Hopper, Ampere, B200, or other Blackwell variants without a rebuild. Do not generalize these flags to other GPUs.

---

## TL;DR for agents (60 seconds)

- **Model**: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` — `Qwen3_5MoeForConditionalGeneration`, **compressed-tensors** NVFP4, ~22 GB. Sparse MoE (256 experts × 8 active + 1 shared, 40 layers), hybrid GatedDeltaNet + full-attention, **multimodal (image + video)**. Abliterated ("heretic") — uncensored.
- **Drafter**: `z-lab/Qwen3.6-35B-A3B-DFlash` — 8-layer, **all-full-attention**, `num_speculative_tokens: 11`.
- **Image**: `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (= `:2026-06-18-v0.23.0-dflashfix`; rollback `:2026-06-11-pr41703`). vLLM 0.23.0 built from source for GB10.
- **It works**: vision/multimodal is functional (a tensor-prefix bug was fixed 2026-06-18 — see below). Scales to **c=64 with zero crashes**; peak ~844 tok/s aggregate.
- **The canonical serve command is at the bottom of this file.** Copy it exactly.

---

## DO NOT UNDO — common stale-documentation traps

### 1. Do NOT force the Marlin NVFP4 MoE backend
```bash
# WRONG — looks reasonable from 2024 Blackwell forum advice:
-e VLLM_TEST_FORCE_FP8_MARLIN=1
```
This model's **256-expert × top-8 MoE prefers CUTLASS** on GB10 — the default auto-selector picks `VLLM_CUTLASS` NVFP4 and that is faster here. Forcing Marlin is a measured regression for *this* shape. (Marlin is only preferred for the Nemotron-Omni 128-expert shape, not this one.) Leave the env unset.

### 2. Do NOT set `--quantization modelopt`
This checkpoint is **`compressed-tensors`** (llm-compressor NVFP4), not modelopt. Use `--quantization compressed-tensors`. (The 27B `-MTP` siblings are modelopt; do not copy their flag here.)

### 3. Do NOT set `--max-num-batched-tokens` to 32k–64k
```bash
# WRONG — "bigger prefill budget = faster":
--max-num-batched-tokens 65536
```
65536 exceeds the inductor compile-range ceiling (`compile_ranges_endpoints: [32768]`) → eager-prefill fallback + heavy unified-memory pressure at load. Use **`16384`** — it stays in the compiled range, frees several GB of activation, and is throughput-neutral because chunked prefill preserves the full 256k context.

### 4. Do NOT push `--gpu-memory-utilization` above 0.88
vLLM v0.23.0 defaults to 0.92, and generic docs suggest 0.90+. GB10's unified LPDDR5X pool is shared CPU+GPU; anything above ~0.88 page-thrashes. Use **`0.85`** when the LLM is the only GPU service; drop to **`0.70`** if co-locating ASR/TTS/embeddings.

### 5. Do NOT add `--mamba-block-size`
Not needed on the v0.23.0 image (`block_size` is an int upstream now), and this model's DFlash drafter is all-full-attention. Adding it is unnecessary and can confuse the cache config.

### 6. Do NOT set `--kv-cache-dtype` or a drafter `attention_backend`
The non-causal DFlash drafter requires its **default backend / BF16 KV**. Do not add `attention_backend` inside the `--speculative-config`, and do not set `--kv-cache-dtype` — both break drafter acceptance (collapses toward 0%, silent ~6× slowdown).

### 7. Do NOT set `num_speculative_tokens` to 15
n≈**11** is optimal here; acceptance and long-context fidelity fall as n climbs past ~11. n=15 just wastes draft compute.

### 8. Do NOT claim vision is broken / strip the vision tower
Vision was fixed 2026-06-18 (see below) and is validated working. Keep `--limit-mm-per-prompt '{"image":4,"video":2}'` to enable full image + video input.

### 9. Do NOT switch `--attention-backend` off `flash_attn`
DFlash's KV-sharing path is only tested with `flash_attn`. `flex_attention`/`triton_attn` for the body break or slow the drafter path.

### 10. Do NOT downgrade the image or revert to the old `vllm-spark-omni-q36` lineage
`aeon-vllm-ultimate:latest` supersedes the per-revision `vllm-spark-omni-q36` (v1/v1.2/v2) containers. Rollback tag is `:2026-06-11-pr41703`, not the omni-q36 images.

### 11. Do NOT `pip install` into the container
The image is source-built; `pip install vllm` or similar will desync the binary from the AEON sm_121a patches. Rebuild from `AEON-7/vllm-ultimate-dgx-spark` if you need to change vLLM.

---

## Vision fix (2026-06-18) — image input works

The published v2 checkpoint had its 333 ViT tensors mis-nested under `model.language_model.visual.*` (a child of the language model) instead of the sibling `model.visual.*` that `Qwen3_5MoeForConditionalGeneration` expects. vLLM silently **skip-loaded the vision tower** → image inputs returned `!!!!`. Fixed by a header-only rename to `model.visual.*` (weight data byte-identical, no re-quant). Validated 7/7 on an image probe with 0 skip-loads. **If you cloned the weights before 2026-06-18, re-pull.** Confirm at boot: the logs must show **zero** `Parameter visual.* not found in params_dict, skip loading` lines.

---

## Required vLLM serve flags (with rationale)

| Flag | Value | Why |
|---|---|---|
| `--quantization` | `compressed-tensors` | this checkpoint's NVFP4 format (not modelopt) |
| `--max-model-len` | `262144` | full 256k context; KV pool sizes from free memory |
| `--max-num-seqs` | `64` | concurrency cap; ~5–6× full-context concurrency at 0.85 |
| `--max-num-batched-tokens` | `16384` | in the compile-range ceiling; chunked prefill keeps 256k |
| `--gpu-memory-utilization` | `0.85` | LLM-only; ≤0.88 cap on GB10 (0.70 if co-located) |
| `--enable-chunked-prefill` | on | decouples 16k prefill chunk from 256k context |
| `--enable-prefix-caching` | on | safe with DFlash on this image (PR #41703) |
| `--trust-remote-code` | on | Qwen3.5/3.6 multimodal processor |
| `--limit-mm-per-prompt` | `{"image":4,"video":2}` | enables full image + video multimodal |
| `--attention-backend` | `flash_attn` | required for the DFlash KV-sharing path |
| `--reasoning-parser` | `qwen3` | splits thinking into the `reasoning` field |
| `--tool-call-parser` | `qwen3_coder` + `--enable-auto-tool-choice` | tool/function calling |
| `--speculative-config` | `{"method":"dflash","model":"/drafter","num_speculative_tokens":11}` | DFlash spec-decode |

No `VLLM_TEST_FORCE_FP8_MARLIN`, no `--kv-cache-dtype`, no `--mamba-block-size`.

---

## Diagnostics — confirm the stack is healthy

```bash
# 1. container running
docker ps --filter name=qwen36-35b
# 2. booted + no skip-loaded vision (must be empty)
docker logs qwen36-35b 2>&1 | grep "not found in params_dict" | head
# 3. endpoint + the 3 aliases
curl -s localhost:8000/v1/models | python3 -c "import sys,json;print([m['id'] for m in json.load(sys.stdin)['data']])"
#    -> ['qwen36-35b-heretic', 'qwen36-fast', 'qwen36-deep']
# 4. DFlash actually firing (after a request)
curl -s localhost:8000/metrics | grep -E "spec_decode_num_(draft|accepted)_tokens_total"
#    accepted/draft ~0.45-0.60 on structured prompts, ~0.21-0.28 on open prose
```

Client note: this is a **reasoning** model — read `choices[0].message.content` **OR** `.reasoning` (thinking is emitted in the `reasoning` field; reading only `content` can look "empty").

---

## Performance baseline (GB10, n=11, measured 2026-06-21)

- **Single-stream (c=1):** 70–142 tok/s decode (Math/Extraction ~140 highest acceptance; Prose/Natural ~70–85 lowest). TTFT ~100–130 ms, TPOT 6.5–14 ms.
- **Aggregate @ c=64:** structured (code/math/reasoning/extraction) **755–844 tok/s** (peak **844**, Reasoning); creative text **497–536** (acceptance-driven). **Zero errors at every concurrency level (1→64).**
- **DFlash acceptance:** ~45–60% structured, ~21–28% creative; holds ~40–45% at 16–20k-token context (the drafter is all-full-attention, so no long-context collapse).
- No stock vanilla-vLLM baseline is published yet, so no stock-vs-optimized multiple is quoted.

---

## Canonical command

```bash
docker run -d --name qwen36-35b --gpus all --ipc host --network host \
  -v ./aeon-model:/model:ro \
  -v ./aeon-drafter:/drafter:ro \
  -v ./vllm-cache:/root/.cache/vllm \
  --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest serve /model \
  --served-model-name qwen36-35b-heretic qwen36-fast qwen36-deep \
  --host 0.0.0.0 --port 8000 \
  --quantization compressed-tensors \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":4,"video":2}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --attention-backend flash_attn \
  --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":11}'
```

The `-v ./vllm-cache:/root/.cache/vllm` mount is optional but recommended — it persists the FlashInfer NVFP4 autotuner + CUDA-graph capture so restarts drop from ~10–12 min (first boot) to ~3–5 min. First boot is slow regardless; that is normal, not a hang.

For the full deployment guide, docker-compose, and per-flag rationale see the repo README. The image is built from `AEON-7/vllm-ultimate-dgx-spark`.
