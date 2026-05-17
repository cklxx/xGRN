# xGRN Mac Performance Plan

Scope: only the official GRN T2I/T2V checkpoints, HBQ tokenizer, and UMT5 text encoder. Hardware target: Apple M4 Pro, 20 GPU cores, Metal 4, 48 GB unified memory, ~273 GB/s. MLX 0.31.2.

## 1. Verified baseline (warm repeat-5 median)

| Profile | End-to-end | GRN refine | Decode | Max RSS | Notes |
|---|---:|---:|---:|---:|---|
| `debug` 0.06M, 2 steps | **1.07 s** | 0.78 s | 0.28 s | 9.91 GB | 102× vs PyTorch/MPS fp32 |
| `t2i-correct` 0.25M, 50 steps | **80.90 s** | 77.70 s | 1.17 s | 9.90 GB | CLIP 0.9634, 5/5 |
| `t2v-short` 0.06M, 16 steps | 17.15 s | 15.85 s | 1.14 s | 9.95 GB | — |
| PyTorch/MPS fp32 `debug` (reference) | 109.19 s | n/a | n/a | 19.37 GB | fp16 path crashes on MPS |

- Default runtime: bf16 UMT5 text encoder, fp32 prompt cache, MLX GRN transformer with `mx.compile` over the fixed-shape CFG visual pass, native MLX HBQ decoder, fp32 GRN safetensors with bf16 compute.
- Correctness gate: `xgrn-parity --full-step` exact, `xgrn-hbq-parity` mean abs `~2.7e-4`, `xgrn-validate` CLIP positive >= 0.93. `0.06M` remains debug-only.

## 2. Where time goes at 0.25M / 50 steps

Per-step GRN cost ≈ **1.554 s** (77.70 s / 50 steps).
- 28 transformer blocks × CFG-batched visual pass + sampling/mask update.
- ≈ 200 Metal kernel dispatches per refinement step.
- GPU utilization ≈ 6%. The dominant cost is **command-buffer submission overhead**, not raw compute or HBM bandwidth.

Root cause: each block fires ~7 separate matmul/norm/rope/sample dispatches; M4 Pro family-9 submission floor is ~50–150 µs per dispatch, so 200×~100 µs ≈ 10–30 ms of pure submission per step is plausible. Every previously verified win (visual-pass compile, native HBQ, bf16 compute) reduces dispatch count or dispatch-side dtype thrash. Every verified regression (per-block compile, mx.async_eval, fused SwiGLU in pure MLX, mx.addmm rewrites) invalidated or fragmented the existing fused graph.

## 3. Strict correctness gate

Before AND after every accepted change:

```bash
uv run python -m py_compile $(git ls-files '*.py')
uv run xgrn-parity --full-step
uv run xgrn-hbq-parity --pt 1 --ph 16 --pw 16
uv run xgrn-hbq-parity --pt 2 --ph 16 --pw 16
uv run xgrn-validate outputs/bench/t2i-correct/latest_t2i.png   # CLIP positive >= 0.93
uv run xgrn-bench --profile debug --warmup --repeat 5            # GRN median <= 0.82 s
```

Any change that fails any gate is reverted and recorded under "Regressed experiments" in README.md, not promoted to a default flag.

## 4. Frontier optimization tracks

Software-only optimization on the current MLX surface is saturated. Remaining wins require either custom Metal kernels, a compile-pass restructuring, or an algorithmic change that preserves the CLIP gate. Tracks are ranked by expected dispatch-reduction × per-dispatch latency floor.

### Track A — Custom Metal kernel fusion (`mx.fast.metal_kernel`)

Each fused kernel must (i) match the existing block dtype contract (fp32 inputs allowed, bf16 compute, fp32 accumulate where MLX already does it), (ii) be drop-in within the compiled visual pass without breaking the fixed-shape kernel cache, (iii) ship with a numerical-parity check vs the current MLX path.

| Fusion | Dispatch collapse | Expected GRN per-step saving | Status |
|---|---|---:|---|
| `rmsnorm + qkv_proj + rope` (one kernel, per block) | 3–4 → 1, ×28 blocks | 30–40 ms | proposed (Track A1) |
| Fused gate+up matmul → SwiGLU split → down (per block) | 4 → 2, ×28 blocks | 15–25 ms | partial flag exists (`--fuse-gate-up`) — make default, audit graph topology |
| `attn_out + residual + rmsnorm_pre_mlp` (per block) | 3 → 1, ×28 blocks | 10–15 ms | proposed (Track A2) |
| `categorical_sample + token_update + mask_schedule` (per step) | 5–10 → 1 with `atomic_outputs=True` | 5–10 ms | proposed (Track A3) |
| `mx.fast.scaled_dot_product_attention` | already fused | — | **do not rewrite** |

Aggregate target: dispatches **200 → 60–80**, GRN per-step **1.55 s → 0.95–1.05 s**, `0.25M/50` GRN **77.70 s → ~50 s** end-to-end **~55–60 s**, a 1.3–1.4× speedup while keeping CLIP >= 0.93.

**Status after 2026-05-15 work:** only **A1-lite** (fused `apply_rope`) delivers a real win at -1.63% wall on `t2i-correct`. **A2** (fused residual + post-attn rms_norm) measured -0.28% GRN with +7% RSS — within noise, kept opt-in only. **A1-full** POC (fused rms_norm + q_proj with naive scalar matmul) is 8× slower than MLX's tuned GEMM in microbench. **A3** (fused sampling) deferred — per dispatch-count theory, expected -0.05% to -0.15%, deep in noise.

**Simdgroup matmul project (Track A1-full real foundation)** — landed in `xgrn_mlx/simdgroup_matmul.py`. Through 10 iterations the kernel went from 8× slower than MLX to **matching/beating MLX**:

| Iteration | Layout | fp32 vs MLX matmul |
|---|---|---:|
| Naive scalar (from earlier commit) | one thread per output element | 8.07× |
| Naive 1-simdgroup-per-tile | each simdgroup does one 8×8 output | 4.39× |
| 2×2 simdgroup layout | 1 accumulator per simdgroup | 2.13× |
| 4×4 simdgroup layout | 1 accumulator per simdgroup | 2.22× |
| 2×2 sg + threadgroup-cached A | manual cooperative load | 3.32× (barriers dominate) |
| 2×2 sg + 2 acc per sg | output-stationary register tiling | 1.82× |
| 2×2 sg + 2×2 register tile (4 acc) | | 1.34× |
| **2×2 sg + 2×4 register tile (8 acc)** | **production layout** | **1.05× fp32, 0.975× bf16** |
| 2×2 sg + 4×2 register tile (8 acc) | other 8-acc shape | 1.09× |
| 2×2 sg + 4×4 register tile (16 acc) | too many registers | 6.32× (spill) |
| 2×2 sg + 4×3 register tile (12 acc) | too many registers | 5.65× (spill) |

The winner — 2×4 register tile per simdgroup (8 accumulators) — packs enough output area per simdgroup to amortize the inner-loop overhead while staying within the M4 Pro register file. **At bf16 the kernel is ~2.5% faster than MLX's bf16 matmul**, and at fp32 within 5%. Byte-exact parity (fp32) / within precision (bf16).

`simd_async_copy` for double-buffered loads was attempted but requires Metal 3's `simdgroup_event` header which is not present in MLX's current toolchain. Deferred until the toolchain catches up.

Steps after the matmul-kernel-itself converged:
1. **M3+M5 (attempted, measured negative)** — wired `--fuse-qkv-metal` to route the 3 qkv projection matmuls through `simdgroup_matmul_bf16` in `_qkv_fused_bf16`. On t2i-correct strict gate: GRN +5.3 %, end-to-end +8.9 %, decode +88 % (!), CLIP 0.9904 → 0.9206. Decoder timing should be untouched by qkv; the regression points to the same integration tax A2 hit — an opaque Metal kernel inside the `mx.compile`'d visual pass breaks fusion/scheduling MLX's native matmul gets. Flag landed but stays opt-in only.
2. **Real M3 (deferred, multi-session)** — fuse rms_norm into the simdgroup matmul kernel via threadgroup-memory phase-1 ssq reduction. Currently blocked by the same integration issue as M5 above — the kernel needs to either be re-expressed as an MLX op the compiler understands, or shipped via an MLX C++ extension (`mlx/extensions`) so it gets first-class treatment in the compile graph.

**Track A summary:** the non-matmul fusion subspace is saturated. The simdgroup matmul project is now at the threshold where M3 fusion can deliver real wall reduction.

### Where time goes inside `visual_forward` (per-op profile, 2026-05-15)

After M3/M5 saturated on integration, ran a non-compiled per-op profile (`outputs/track-a/profile/block_ops.py`) to find new targets:

| Category | ms / step | % | Notes |
|---|---:|---:|---|
| MLP `down_proj` matmul | 326 | 18.7 % | [B*L, 8192] @ [8192, 2304] |
| MLP `up_proj` matmul | 323 | 18.5 % | [B*L, 2304] @ [2304, 8192] |
| MLP `gate_proj` matmul | 322 | 18.5 % | [B*L, 2304] @ [2304, 8192] |
| SDPA (`mx.fast.scaled_dot_product_attention`) | 171 | 9.8 % | already a fused Apple kernel |
| QKV matmuls (q+k+v+o) | 388 | 22.4 % | M3/M5 target — saturated |
| RoPE q+k (after A1-lite) | 77 | 4.4 % | A1-lite already saved ~50 % of this |
| silu(gate)*up + small ops | 68 | 3.9 % | A2's class of work |
| Everything else | 137 | 7.8 % | residuals, norms, transposes |

**Headline:** matmuls in total are **78 %** of per-step time. The MLP block alone is **56 %** — 2.5× bigger than the QKV block we just spent two days on. Reducing matmul count or matmul cost is the *only* meaningful lever; everything else is in the noise floor.

Attempted MLP fusion via `--fuse-mlp-gate-up` (existing flag, MLX matmul on concatenated gate+up weight): regressed t2i-correct by +5.17 % GRN / +8.67 % wall with **CLIP bit-identical** (0.9904 exact). Same regression pattern as `--fuse-qkv-concat`: manual Python-level matmul fusion defeats whatever the mx.compile pass is already doing on the 2 separate matmuls.

**Conclusion:** matmul fusion via `mx.fast.metal_kernel` regresses (integration tax). Matmul fusion via manual MLX concatenation regresses (compile already smarter). The only remaining lever is to either (a) rewrite `simdgroup_matmul` as an MLX C++ extension under `mlx/extensions/` so the compile graph treats it as a first-class primitive, or (b) accept that Track A is saturated on this MLX 0.31 / Metal 4 / M4 Pro stack and the next perf frontier is somewhere other than matmul fusion (training-time distillation per Track D, or architectural changes).

Family-9 implementation rules for every kernel:
- Use `simdgroup_matrix_storage` 8×8×8 bf16 multiplies for any inner-loop matmul.
- Specialize via `mx.fast.metal_kernel` template params on `head_dim` and `hidden_dim` for compile-time unrolling.
- Keep registers ≤ 32/thread and threadgroup memory ≤ 16 KB to retain occupancy.

### Track B — Whole-block `mx.compile(shapeless=True)`

Per-block compile already regressed. The unexplored direction is a **single shapeless compile over the full 28-block GRN forward**, pinning constants via `inputs=[...]`. This lets MLX's fusion pass merge across block boundaries (block-N residual + block-(N+1) pre-norm + qkv weight read), which per-block compile prohibits. Must not break the existing visual-pass compile — the two compiles are nested; verify with a single warm step that `compiled_graph_hits` is constant across steps 2–50.

### Track C — Late-step-only CFG and KV-cache sharing

Algorithmic, not kernel. Published finding for MaskGIT-style discrete masked generators: high CFG weight early in sampling hurts quality, late guidance dominates effect. Tactic: run only the conditional visual pass for steps 0..K-1 and the CFG-batched cond+uncond pass for steps K..49. Implemented as `--cfg-start-step K`, default `K=0` (bit-identical to today). When `step < K`, the loop runs the existing `visual_forward_embedded(visual_input, rope, cond_cache)` path (B=1) and skips the CFG mix.

Sweep on the standard `t2i-correct` prompt (0.25M / 50 steps, `--stable-shape-warmup --repeat 3`):

| K | end_to_end (s) | CLIP positive | passes 0.93 | save vs K=0 |
|---:|---:|---:|:---:|---:|
| 0 | 76.66 | 0.9904 | yes | — |
| 5 | 72.89 | 0.8235 | NO | +4.9 % |
| 10 | 69.44 | 0.9332 | yes (razor) | +9.4 % |
| **15** | **66.10** | **0.9635** | **yes** | **+13.8 %** |
| 17 | 64.66 | 0.5997 | NO | +15.7 % |
| 20 | 62.67 | 0.7440 | NO | +18.2 % |
| 25 | 59.48 | 0.0028 | NO | +22.4 % |

Wall savings scale roughly linearly with K (~1 s per K-step, consistent with cond-only visual forward costing about half of the CFG-batched two-lane forward). Quality is highly non-monotonic in K — K=15 passes solidly, K=17 collapses, K=20 partially recovers, K=25 fully collapses. Not the smooth quality curve the published MaskGIT late-CFG result predicts for our GRN-2B / 50-step / 0.25M setup.

Decision: shipped as opt-in `--cfg-start-step K`, default K=0, **no default change**. K=15 is the recommended user-tunable speed knob on the standard prompt.

A partial multi-prompt re-validation (orange_tabby only, with harder CLIP distractors `"random noise pattern"`, `"a black and white photograph"`) revealed two important things before the sweep was stopped to avoid concurrent GPU pressure:

1. **The 0.93 gate was an artifact of the original validator's specific 4-negative set.** Against harder distractors, the same K=0 image scored `0.685` instead of `0.9904`. The strict 0.93 threshold cannot be re-applied to a different negative set without recalibration.
2. **K=10 is not actually safe.** The K=10 image's top CLIP label flips to `"a blurry distorted image"` against the harder negatives — real, measurable quality cost that the original easy negative set hid. K=15 keeps the correct top label on this single prompt.

These findings sharpen the recommendation: K=10 is *not* a safer middle ground, and any future default promotion of K=15 needs a multi-prompt sweep with prompt-specific positives, harder shared distractors, and a recalibrated gate (loose: top ∈ positives ∧ positive > negative ∧ positive ≥ 0.5; promote default only if every prompt clears it at both K=0 and K=15, and K=15 stays within 0.10 of K=0 per-prompt).

Sub-tactic still proposed (not yet implemented): cache the visual-tower KV that does not depend on prompt text across CFG lanes so the uncond lane only pays the cross-attn delta. The cond-only branch also uses uncompiled `block_maybe_compiled`; compiling it for B=1 fixed shape would tighten the savings curve and remove the +0.8 GB RSS overhead during steps < K. Both belong to a future iteration.

### Track D — Step distillation as a future training run

Out-of-scope for the current runtime but recorded as the highest-ceiling remaining win. DiMO (ICCV 2025) and CDLM (arxiv 2511.19269) report 3.6–14.5× step reduction for discrete masked predictors via consistency-style self-distillation, all preserving CLIP-comparable quality. 50 → 8–12 steps would crush every kernel-level win combined. Requires offline training infra; tracked but not started.

### Track E — Profiling to confirm the dispatch hypothesis

Before any A1/A2 kernel work lands, capture and inspect once:

```bash
MTL_CAPTURE_ENABLED=1 uv run python - <<'PY'
import mlx.core as mx
from xgrn_mlx.run import generate_mac
mx.metal.start_capture("outputs/profile/grn_step.gputrace")
generate_mac(task="T2I", prompt="orange cat", seed=42, pn="0.25M", steps=2)
mx.metal.stop_capture()
PY
```

Open in Xcode → Metal Debugger → Dependencies view. Report:
- command-buffer count for one refinement step,
- mean inter-dispatch CPU gap,
- per-dispatch GPU-active time vs wall time.

**Decision rule.** Mean gap > 50 µs ⇒ dispatch-bound, ship Track A. Mean gap < 20 µs but per-dispatch occupancy low ⇒ kernel-size problem, prioritize `simdgroup_matrix_storage` rewrites over fusion. Mean gap < 20 µs and occupancy high ⇒ MLX runtime is near hardware limit; only Tracks C and D remain.

## 5. Saturation stop rule

- Any change that does not improve `t2i-correct` warm repeat-5 median end-to-end by >= 2% with CLIP positive >= 0.93 is kept only as an opt-in flag, never as a default.
- A change that improves the gate but regresses `debug` warm median by > 5% is rejected on the dispatch-overhead grounds it implies a graph-cache break.
- Every accepted optimization must preserve `xgrn-parity --full-step` exact and `xgrn-hbq-parity` mean abs <= `5e-4`.
- README speed table and "Experiment Outcomes" table are updated in the same commit as the runtime change.

## 6. Active codex tracks

| Track | File | Owner |
|---|---|---|
| Track A1: fused rmsnorm+qkv+rope Metal kernel | `xgrn_mlx/grn.py` | codex tmux `codex-a` |
| Track B: shapeless whole-stack `mx.compile` | `xgrn_mlx/grn.py`, `xgrn_mlx/run.py` | codex tmux `codex-b` |
| Track C: late-step-only CFG with K sweep | `xgrn_mlx/grn.py`, `xgrn_mlx/run.py`, `xgrn_mlx/bench.py` | codex tmux `codex-c` |

Each tmux session is started by this repo's owner and rotated independently. Track owners are responsible for running the strict correctness gate before requesting promotion to default.

## 7. Reference

- Custom Metal kernels in MLX: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX compilation and `mx.compile(shapeless=True)`: https://ml-explore.github.io/mlx/build/html/usage/compile.html
- MLX Metal debugger: https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html
- WWDC25 Metal 4: https://developer.apple.com/videos/play/wwdc2025/205/
- WWDC25 LLMs on Apple Silicon with MLX: https://developer.apple.com/videos/play/wwdc2025/220/
- Self-Guidance for MaskGIT (late-stage CFG): https://arxiv.org/html/2410.13136v1
- DiMO step distillation (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_DiMO_Distilling_Masked_Diffusion_Models_into_One-step_Generator_ICCV_2025_paper.pdf
- CDLM consistency distillation for discrete masked LMs: https://arxiv.org/pdf/2511.19269

## 8. 2026-05-17 Deep Research Sweep — All 7 Spikes Negative

Four parallel research agents (framework gaps / external projects / Track D distillation / architectural prototypes) produced a Top-7 prioritized spike list. **All seven tested negative** on `t2i-correct` warm. Only `P0-3` (cache fp32-cast norm weights) was shipped at ~0.1% step savings. The negative results pin down GRN's saturation profile precisely and rule out a class of "obvious" follow-ups.

### Calibrations from the research

- M4 Pro peak measured **7.59 TFLOP/s bf16** (we'd been using 6.0 in earlier analysis). MLP matmul utilization is therefore **74–95% of peak**, not 87–95%. Real but smaller compute headroom than v3 diagram implied.
- xGRN is double-outlier vs industry: BF16 instead of FP16 (Draw Things/mflux/DiffusionKit/Apple all FP16), 50 steps instead of 8–24 (MaskGIT 8, Muse 24, Sana 14–20).
- Apple GPU shader cores **do not** have `simdgroup_matrix<int8>` — `simdgroup_matrix` only supports fp16/bf16/fp32. int8 tensor ops are ANE-only. This is the hardware-level explanation for the 2026-05-15 quant-matmul regression.
- Most diffusion distillation literature (Progressive Distillation, Consistency Models, LCM, DMD, ADD/Turbo) requires a PF-ODE or score function on continuous latents. GRN's categorical discrete refinement on continuous `pt` AdaLN cannot use any of them. **Discrete-diffusion family (DiMO, Di4C, SDTT, CDLM/MPDC) is the only viable distillation toolset.**

### Spike results (all measured on `t2i-correct` warm, R=2 unless noted)

| # | Spike | Wall delta | CLIP delta | Status | Root cause |
|---|---|---:|---:|---|---|
| P0-2 | `mx.compile(shapeless=True)` on visual_pass | n/a | n/a | RED | `Slice cannot infer output shapes` — KV-cache concat in graph needs static shape |
| P0-3 | Cache fp32-cast norm weights | ≈0 | none | **GREEN, shipped** | Eliminates 7000 redundant astype dispatches; gain is noise but cost is zero |
| P1-3 | Swap A1-lite fused_rope → `mx.fast.rope` | n/a | n/a | RED | GRN uses 3D RoPE (frame × height × width); `mx.fast.rope` is 1D-sequence-offset only |
| P1-4 | `MLX_SDPA_BLOCKS=32` (M4-Pro tuning) | -0.3% | none | YELLOW | Isolated SDPA -30% but in `mx.compile` pipeline it's already overlapped |
| P0-1 | `compute_dtype=fp16` end-to-end | **+1.6%** | none | RED | Isolated small-shape matmul -7.5% but pipeline cast overhead dominates |
| P1-1 | SVD low-rank `down_proj` R=1536 | -1.7% | **0.99 → 0.01** | RED | GRN MLP weights are high-rank (S[0]=78, S[-5]≈2.5); 67% rank drop = 27% Frobenius loss. Even R=2304 (full rank) drops CLIP to 0.69 because bf16 dual-matmul accumulates precision error |
| P1-2 | Drop block 14 (visual_forward) | -3.5% | **0.99 → 0.002** | RED | GRN has no layer-level redundancy; every block is doing essential work |

### What this tells us about GRN as a model

1. **Weights are full-rank**: SVD spectrum S[0]=77.6, S[-5]≈2.5 — flat decay across all 2304 ranks. LLM compression literature (ASVD, SVD-LLM) doesn't transfer because GRN was trained from scratch as a 2B refinement model with no over-parameterization slack.
2. **Layers are non-redundant**: dropping any single mid-network block kills CLIP. Refinement step composition means each block's contribution is load-bearing.
3. **bf16 is mandatory at the storage level**: not just for range, but because dual-matmul (e.g. SVD A,B) breaks under bf16 precision compounding.
4. **mx.compile already at ceiling**: cross-step pipelining (`async_eval`), shapeless, weight pre-transpose, FP16 — all return ≈0 or regressive.

### Updated next-step priority order

The framework + architecture layer is **definitively saturated**. Two paths remain, both training-side or research-grade:

1. **Track D distillation** — **DiMO** (ICCV 2025, Meissonic teacher) is the canonical fit. GRN forward already accepts continuous `pt` AdaLN so the integration path is direct. Cloud-only: 8×A100 · 3-5 days · ~$1.5K–$2.5K for 50→8 steps; +$1.5K for Di4C-augmented 8→4. Expected wall: **76 → 13–25 s e2e**. M4 Pro can host LoRA fine-tuning only, not from-scratch distillation.
2. **DiTFastAttn cross-timestep attention sharing** (arXiv 2406.08552) — training-free, claims -76% attention FLOPs, 1.8× e2e on DiT models. 50 step × 28 block has plausibly high adjacent-step attention similarity. **Untested on discrete-refinement architectures**; needs a 1–2 week prototype. Lower predicted impact than DiMO but no cloud cost.
3. **Token-Critic sampler** (Lezama 2022) — train an external critic, $100 one A100·80h, can drop GRN's 50 → ~16 steps. Sampler-only, no main-model retrain.

### Experimental flags retained (default off, will not be promoted)

These spikes added CLI flags that produce known regressions. They are kept behind `--fuse-...` flags with `default=0/empty` to prevent re-discovery cost:

| Flag | Spike | Why kept | Promotion gate |
|---|---|---|---|
| `--fuse-mlp-lowrank-R N` | P1-1 SVD | Future ASVD-style activation-whitened SVD might unstick this | CLIP positive ≥ 0.93 |
| `--drop-blocks N1,N2,...` | P1-2 layer-drop | Conditional drop (only late steps where signal converged) might be viable | CLIP positive ≥ 0.93 |
| `--fuse-swiglu-metal` | 2026-05-15 | Loses to mx.compile (-56%) but useful reference for kernel authoring | — |
| `--fuse-qkv-*` (7 variants) | 2026-05-15 | All regressed +3.77–7.30%; left as history | — |

### Saturation stop rule (updated 2026-05-17)

Going forward, do not propose framework-layer or single-layer architectural changes unless they include:
- a published paper specifically on **discrete-token / categorical / masked refinement** models (not LLM, not DDPM), **and**
- a measurement plan that pre-quantifies CLIP-gate risk before the wall-time experiment.

Generic LLM compression / DDPM distillation literature has been definitively ruled out by the 2026-05-17 sweep.

## 9. References (research sweep)

- DiMO (ICCV 2025): https://arxiv.org/abs/2503.15457 · code https://github.com/yuanzhi-zhu/DiMO
- Di4C (ICML 2025, Sony): https://arxiv.org/abs/2410.08709 · code https://github.com/sony/di4c
- DiTFastAttn: https://arxiv.org/html/2406.08552v1
- ASVD: https://arxiv.org/abs/2312.05821 · SVD-LLM: https://arxiv.org/abs/2403.07378
- Token-Critic (MaskGIT improvement): https://ar5iv.labs.arxiv.org/html/2209.04439
- CDLM/MPDC: https://arxiv.org/abs/2605.00161
- Apple MSL Spec (simdgroup_matrix limits): https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
