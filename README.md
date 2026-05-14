# xGRN — GRN on Apple Metal, fast

> **102x faster** than stock PyTorch/MPS (warm, debug profile) · **25x faster** cold start · native MLX + Metal

Mac-specialized runtime for the official GRN T2I/T2V models. The GRN transformer
and refinement loop run in MLX. Prompt embeddings are cached as MLX-readable
artifacts after the first UMT5 run. HBQ decode uses a native MLX decoder by
default, with the official PyTorch/MPS decoder kept as a fallback.

## Speed at a glance

| | PyTorch/MPS fp32 | xGRN MLX bf16 (cold) | xGRN MLX bf16 (warm) |
|---|---:|---:|---:|
| `debug` 0.06M, 2 steps | 109.19 s | **4.39 s** | **1.07 s** |
| Speedup vs PyTorch | — | **25×** | **102×** |

Full benchmark table in [Performance](#performance).

## Quickstart

### Install

```bash
uv sync --python 3.11
```

xGRN reuses the official GRN text encoder implementation, so the official
source checkout must sit beside this repo:

```bash
cd ..
git clone https://github.com/MGenAI/GRN GRN
cd xGRN
```

### Start

```bash
uv run xgrn-app
```

Open <http://127.0.0.1:7860>. If the port is busy, use another one:

```bash
uv run xgrn-app --server-port 7861
```

### First Use

On first startup, `xgrn-app` checks `models/GRN`. If the required weights are
missing, it downloads the official HuggingFace snapshot with progress, retries a
failed download once, then creates the MLX artifacts used by the app. Existing
files are reused, so later launches skip the download and conversion.
The full T2I+T2V cache is large; keep enough free disk for the raw HuggingFace
snapshot plus MLX `fp16`/`fp32` artifacts.

T2I and T2V weights are prepared by default. The cache location, model id, and
revision are configurable:

```bash
XGRN_MODEL_DIR=/Volumes/ssd/xgrn/GRN \
XGRN_HF_REPO_ID=bytedance-research/GRN \
XGRN_HF_REVISION=main \
uv run xgrn-app --server-port 7861
```

Set `XGRN_AUTO_DOWNLOAD=0` or pass `--no-auto-download` when you want startup to
fail fast with a repair command instead of downloading.

Manual prefetch is optional, but useful before a demo or on a slow network:

```bash
uv run xgrn-download --model-dir models/GRN
uv run xgrn-run --task T2I
uv run xgrn-validate outputs/latest_t2i.png
```

If startup is blocked, the error prints the missing files and the exact command
to repair the cache.

## Outputs

| File | Description |
|---|---|
| `outputs/latest_t2i.png` | Most recent T2I result |
| `outputs/latest_t2v.mp4` | Most recent T2V result |
| `outputs/latest_refinement.gif` | Per-step frames (when capture enabled) |
| `outputs/refinement_stats.csv` | Per-step timing stats |

## Defaults

| Setting | Value |
|---|---|
| Token budget | `0.25M` (correctness gate) |
| Refinement steps | 50 |
| Temperature | 1.1 |
| Text encoding | bf16 UMT5 |
| Prompt embedding cache | fp32 |
| GRN matmul compute | bf16 |
| Visual pass | fixed-shape compiled CFG |
| HBQ decoder | native MLX |

`0.06M` is available as a fast debug preset but is not accepted as the semantic
correctness gate. The runtime supports higher step counts (150, 250); latency
scales roughly linearly.

**Prompt cache:** the first run for a new prompt builds the embedding cache.
Subsequent runs with the same prompt skip that cost entirely. Benchmark rows
report `end_to_end_sec` as the primary metric (includes generation, decode,
output writing, and validation). `generation_wall_sec` is kept as a no-validation
comparison point.

**Warm benchmarks:** use `--warmup --repeat 5` for full warm-run timing. For
long profiles, `--stable-shape-warmup` warms the same MLX graph, KV cache, GRN
weights, and decoder weights with two refinement steps before measuring the real
profile.

**Low-memory cleanup:** `xgrn-run` and `xgrn-bench` support
`--release-after-run`; add `--release-text-cache` to also clear in-process prompt
embedding arrays. This trades away repeated-prompt warm-cache speed for lower
post-run memory pressure.

**Validation artifact:**

```bash
uv run xgrn-validate outputs/orange_tabby_50steps_seed42_pn025M_bf16text.png
```

## Performance

All timings are strict end-to-end wall time. Rows with validation include CLIP
time in `end_to_end_sec`.

| Profile | Runtime | Command | End-to-end | GRN refine | Decode | Max RSS | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `debug` (0.06M, 2 steps) | PyTorch/MPS fp16 | `uv run xgrn-pytorch-bench --profile debug` | failed | failed | failed | n/a | MPS matmul dtype assertion |
| `debug` (0.06M, 2 steps) | PyTorch/MPS fp32 | `uv run xgrn-pytorch-bench --profile debug --device mps --dtype fp32` | 109.19 s | n/a | n/a | 19.37 GB | `outputs/pytorch_bench/debug-mps-fp32-report.json` |
| `debug` (0.06M, 2 steps) | **xGRN MLX bf16 + native HBQ cold** | `uv run xgrn-bench --profile debug` | **4.39 s** | 3.81 s | 0.56 s | 9.91 GB | `outputs/bench/debug-native-decoder-bf16-cold-report.json` |
| `debug` (0.06M, 2 steps) | **xGRN MLX bf16 + compiled visual + native HBQ warm median** | `uv run xgrn-bench --profile debug --warmup --repeat 5` | **1.07 s** | 0.78 s | 0.28 s | 9.91 GB | `outputs/bench/debug-compile-visual-pass-warm-repeat5-report.json` |
| `debug` (0.06M, 2 steps) | xGRN MLX bf16 + native HBQ warm, no visual compile | `uv run xgrn-bench --profile debug --no-compile-visual-pass --warmup --repeat 5` | 1.18 s | 0.86 s | 0.31 s | 8.36 GB | fallback comparison |
| `debug` (0.06M, 2 steps) | xGRN MLX fp32 + MPS HBQ warm median | `uv run xgrn-bench --profile debug --compute-dtype fp32 --decoder-backend mps --warmup --repeat 5` | 1.80 s | 0.92 s | 0.87 s | 13.97 GB | old fallback comparison |
| `t2i-correct` (0.25M, 50 steps) | xGRN MLX bf16 + MPS HBQ + CLIP | `uv run xgrn-bench --profile t2i-correct --decoder-backend mps` | 103.67 s | 89.24 s | 8.13 s | 13.73 GB | CLIP score 0.9618 |
| `t2i-correct` (0.25M, 50 steps) | **xGRN MLX bf16 + compiled visual + native HBQ + CLIP warm median** | `uv run xgrn-bench --profile t2i-correct --stable-shape-warmup --repeat 5` | **80.90 s** | 77.70 s | 1.17 s | 9.90 GB | CLIP 5/5, score 0.9634 |
| `t2i-correct` (0.25M, 50 steps) | low-memory fp16 weights + bf16 compute + native HBQ + CLIP | `uv run xgrn-bench --profile t2i-correct --weights-dtype fp16` | 89.87 s | 82.47 s | 1.27 s | 6.31 GB | CLIP score 0.7247 |
| `t2i-step-stress` (0.06M, 150 steps) | xGRN MLX bf16 + compiled visual + native HBQ + CLIP | `uv run xgrn-bench --profile t2i-step-stress --stable-shape-warmup` | 67.04 s | 59.87 s | 0.31 s | 10.48 GB | CLIP failed, score 0.0010 |
| `t2v-short` (0.06M, 16 steps) | xGRN MLX bf16 + compiled visual + native HBQ | `uv run xgrn-bench --profile t2v-short` | 17.15 s | 15.85 s | 1.14 s | 9.95 GB | `outputs/bench/t2v-short-default-compile-visual-bf16-native-report.json` |

For the `debug` profile: **24.9× faster cold start** and **102.3× faster warm**
vs original PyTorch/MPS fp32.

### Where time goes at `0.25M / 50 steps`

Per-step GRN ≈ **1.55 s** (77.70 s / 50). 28 transformer blocks × CFG-batched
visual pass + sampling/mask update fire ≈ 200 Metal kernel dispatches per step.
GPU utilization ≈ 6 %. The dominant cost on M4 Pro is command-buffer submission
overhead (~50–150 µs per dispatch), not raw compute or HBM bandwidth.

That single fact decides what we try next:

| Track | What | Expected | Status |
|---|---|---:|---|
| A1 | Custom `mx.fast.metal_kernel`: fused `rmsnorm + qkv_proj + rope` per block | ~30–40 ms/step | active, codex tmux |
| A2 | Custom Metal: fused `attn_out + residual + pre-MLP rmsnorm` | ~10–15 ms/step | proposed |
| A3 | Custom Metal: fused sampling + mask update with `atomic_outputs` | ~5–10 ms/step | proposed |
| B | Whole-stack `mx.compile(shapeless=True)` across all 28 blocks | cross-block fusion | active, codex tmux |
| C | Late-step-only CFG (`--cfg-start-step K`, skip uncond before step K) | 13.8 % wall at K=15 on the standard prompt | shipped opt-in, K=0 default, see Experiment Outcomes |
| D | Step distillation (DiMO / CDLM) → 8–12 steps | 3–10× | future training |

Aggregate ceiling on the kernel-fusion tracks alone: dispatches **200 → ~70**,
GRN per-step **1.55 s → ~1.0 s**, `0.25M/50` end-to-end **80.90 s → ~55 s**.
Every track must keep `xgrn-parity --full-step` exact and CLIP positive
`>= 0.93`. See `PERFORMANCE_PLAN.md` for the full rules and `CODEX_TASKS.md`
for the active task briefs.

### Experiment Outcomes

| Experiment | Result |
|---|---|
| Per-block `mx.compile` | regressed debug warm to 1.97 s |
| `mx.async_eval` in refinement loop | regressed debug warm to 2.34 s |
| `--compile-cfg-logits` | passed CLIP at 90.03 s / 8.87 GB RSS, but slower than default |
| `--compile-refinement-update` | fixed-shape compiled sampling/mask update regressed debug warm to 1.10 s |
| `--sampling-mode argmax` | debug warm median 1.09 s, not faster than stochastic categorical; kept for debug only |
| `--sampling-mode binary` | Bernoulli sampler for the fixed two-class logits; debug warm repeat-3 regressed to 1.13 s |
| `--linear-quantization int8` | passed CLIP but slowed `0.25M/50` to 105.42 s; not useful vs fp16 weights |
| `--linear-quantization int4` | debug warm median 1.09 s, no speed or RSS win |
| `--weights-dtype fp16` | passed CLIP at 89.87 s / 6.31 GB RSS, lower semantic margin; kept as low-memory mode |
| `--text-dtype fp16 --text-cache-dtype fp16` | loose validator passed, but strict CLIP gate missed (`0.8493 < 0.93`) and GRN slowed to 86.33 s |
| `--mask-schedule dus` | 20/30/40 steps were faster but failed CLIP; 40 steps reached only 0.3683 positive score |
| no-upcast RMSNorm experiment | regressed the default debug gate; fp32 `mx.fast.rms_norm` remains default |
| `--precompute-pt-embed` | debug warm repeat-3 regressed to 1.09 s; default remains on-demand |
| `--fuse-swiglu-metal` | numeric diff `2.38e-7`, but debug warm repeat-3 regressed to 1.11 s |
| `--stack-cfg-cache` | reduces Python arguments but debug warm repeat-3 regressed to 1.09 s |
| `--min-change-frac 0.005` | did not early-stop on a 0.06M/20 smoke; not a default speed path |
| `--track-token-confidence` | `0.25M/50` trace shows 50% of tokens exceed 0.9 confidence only at step 38, so sparse DUS is not justified yet |
| `--cfg-start-step K` (Track C) | K=0 byte-identical to today. On the standard `t2i-correct` prompt: K=10 passes razor-thin (CLIP 0.9332, 9.4% saving), **K=15 passes with margin (0.9635, 13.8% saving, end-to-end 76.66 → 66.10 s)**, K=17/20/25 collapse CLIP (0.5997/0.7440/0.0028). Quality non-monotonic in K → ship opt-in, do not flip default until a multi-prompt K stability sweep clears every prompt at 0.93. |

Regressed experiments remain as opt-in flags for investigation. `--no-compile-visual-pass`,
`--compile-refinement-update`, `--sampling-mode argmax`, `--sampling-mode binary`,
`--linear-quantization`, `--mask-schedule dus`, `--precompute-pt-embed`, `--fuse-swiglu-metal`,
`--stack-cfg-cache`, `--compute-dtype fp32`, `--decoder-backend mps`, and `--cfg-start-step` are kept
for parity and debug comparison. `--weights-dtype fp16` is the strongest
low-memory correct path, but fp32 weights stay default for the best speed and
semantic margin.

See `PERFORMANCE_PLAN.md` for the verified baseline and optimization roadmap,
and `CODEX_TASKS.md` for the active Codex task briefs (Tracks A/B/C above).

## Project status

Software-only MLX optimization for the current runtime is saturated. Remaining
wins require custom Metal kernels (Track A), a compile-pass restructuring
(Track B), or algorithmic changes that preserve the CLIP gate (Tracks C/D).
Three independent codex tmux sessions own one track each — they share the
strict correctness gate defined in `PERFORMANCE_PLAN.md` §3 and are not
permitted to flip defaults without passing it.
