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

Family-9 implementation rules for every kernel:
- Use `simdgroup_matrix_storage` 8×8×8 bf16 multiplies for any inner-loop matmul.
- Specialize via `mx.fast.metal_kernel` template params on `head_dim` and `hidden_dim` for compile-time unrolling.
- Keep registers ≤ 32/thread and threadgroup memory ≤ 16 KB to retain occupancy.

### Track B — Whole-block `mx.compile(shapeless=True)`

Per-block compile already regressed. The unexplored direction is a **single shapeless compile over the full 28-block GRN forward**, pinning constants via `inputs=[...]`. This lets MLX's fusion pass merge across block boundaries (block-N residual + block-(N+1) pre-norm + qkv weight read), which per-block compile prohibits. Must not break the existing visual-pass compile — the two compiles are nested; verify with a single warm step that `compiled_graph_hits` is constant across steps 2–50.

### Track C — Late-step-only CFG and KV-cache sharing

Algorithmic, not kernel. Published finding for MaskGIT-style discrete masked generators: high CFG weight early in sampling hurts quality, late guidance dominates effect. Tactic: run only the conditional visual pass for steps 0..K-1 and the CFG-batched cond+uncond pass for steps K..49. With K=25 this halves visual-pass cost on the first half ≈ **15–20% wall-time reduction**. Sub-tactic: cache the visual-tower KV that does not depend on prompt text across CFG lanes so the uncond lane only pays the cross-attn delta.

Gate requirement: sweep K ∈ {10, 20, 25, 30} at `t2i-correct` and pick the largest K that holds CLIP positive >= 0.93. If no K passes, demote to an opt-in `--cfg-start-step K` flag, not a default.

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
