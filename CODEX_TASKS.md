# xGRN Codex Tasks — Frontier Tracks

Hardware: Apple M4 Pro, 20 GPU cores, Metal 4, 48 GB unified memory, ~273 GB/s. MLX 0.31.2, `mx.fast.metal_kernel` available.

Root cause we are attacking: at `0.25M / 50 steps`, per-step GRN is **~1.55 s**. Approximately **200 Metal dispatches per step** with **~6 % GPU utilization** ⇒ M4 Pro command-buffer submission overhead dominates. The right lever is **dispatch-count reduction via kernel fusion** (Track A), **cross-block compile fusion** (Track B), and **algorithmic step-cost reduction that preserves CLIP** (Track C).

Three independent codex sessions own one track each. They MUST NOT touch another track's files without coordination.

---

## Hard rules (apply to every track)

1. **Strict correctness gate.** Before AND after every change, run:
   ```bash
   uv run python -m py_compile $(git ls-files '*.py')
   uv run xgrn-parity --full-step
   uv run xgrn-hbq-parity --pt 1 --ph 16 --pw 16
   uv run xgrn-hbq-parity --pt 2 --ph 16 --pw 16
   uv run xgrn-bench --profile debug --warmup --repeat 5 --report /tmp/bench-debug.json
   ```
   `xgrn-parity --full-step` must remain exact. Debug GRN median must not regress by more than 5 % over the 0.78 s baseline (cap 0.82 s). Track A/B promotions also require:
   ```bash
   uv run xgrn-bench --profile t2i-correct --stable-shape-warmup --repeat 5 --report /tmp/bench-correct.json
   uv run xgrn-validate outputs/bench/t2i-correct/latest_t2i.png       # CLIP positive >= 0.93
   ```
2. **Do not break the fixed-shape visual-pass `mx.compile`.** Any change that invalidates its kernel cache (dtype changes, graph topology changes on inputs that flow into the compiled function) regresses on first call AND every call after if shapes drift. Test that step-2 wall time equals step-50 wall time within 5 %.
3. **Do not change `mx.fast.rms_norm` upcast.** `mx.fast.rms_norm(x.astype(fp32), weight.astype(fp32))` is the calibrated fast path. Tested → regresses.
4. **Do not rewrite `mx.fast.scaled_dot_product_attention`.** Already fused.
5. **Do not use `mx.addmm`** to rewrite `embed_visual_labels`. Tested → breaks compile graph topology.
6. **Land behind an opt-in flag first**, default OFF, with a CLI switch and a bench row. Only flip default after Track owner runs the strict gate and reports >= 2 % wall improvement on `t2i-correct` warm repeat-5 median.

---

## Track A — Custom Metal kernel fusion (`mx.fast.metal_kernel`)

**Owner**: codex tmux `codex-a`. Files: `xgrn_mlx/grn.py` (kernel implementations, integration), `xgrn_mlx/parity.py` (numerical-parity check), `xgrn_mlx/bench.py` (CLI flag wiring).

### A1 — Fused `rmsnorm + qkv_proj + rope` (start here)

Each block currently fires 3–4 dispatches just for QKV prep. Fuse them.

**Kernel contract.**
- Inputs: `x` shape `[B, L, H]` (cond+uncond batch B=2 for visual pass), `norm_weight [H]`, `qkv_weight [H, 3*H_kv_total]`, `rope_cos [L, head_dim]`, `rope_sin [L, head_dim]`. All bf16 except norm_weight fp32.
- Outputs: `Q [B, n_heads, L, head_dim]` rotated, `K [B, n_heads, L, head_dim]` rotated, `V [B, n_heads, L, head_dim]` plain.
- Inside the kernel: fp32 RMS normalization (parity with `mx.fast.rms_norm`), bf16 simdgroup-matrix matmul for QKV via `simdgroup_matrix_storage` 8×8×8, RoPE applied only to Q and K.

**Implementation steps.**
1. Land the kernel as `fused_norm_qkv_rope(x, norm_w, qkv_w, cos, sin, n_heads, head_dim)` in `xgrn_mlx/grn.py`, gated by a new `--fuse-attn-prelude` CLI flag (default OFF) and a `fuse_attn_prelude: bool = False` constructor arg on `GRN2BMLX`.
2. Add to `xgrn_mlx/parity.py` a check: random `x`, real block weights, compare `(Q, K, V)` against the current 3-dispatch path. Pass threshold: max abs diff `<= 1e-2` in bf16. Wire as `xgrn-parity --fused-attn-prelude`.
3. Bench gate: `uv run xgrn-bench --profile debug --fuse-attn-prelude --warmup --repeat 5`. Expected GRN median drop from 0.78 s to ~0.74 s on debug.
4. If A1 ships, then A2.

### A2 — Fused `attn_out + residual + pre-MLP rmsnorm`

After A1 ships and stable on `t2i-correct`. Same playbook. Inputs: `attn_out [B, L, H]`, `residual [B, L, H]`, `norm_weight [H]`, `o_proj_weight [H, H]`. Output: `[B, L, H]` ready for MLP. Save 2 dispatches × 28 blocks.

### A3 — Fused sampling + mask update

After A2. Single kernel for the categorical sampler + token write + DUS/random mask update. Use `mx.fast.metal_kernel(..., atomic_outputs=True)` for the in-place token buffer write.

---

## Track B — Shapeless whole-stack `mx.compile`

**Owner**: codex tmux `codex-b`. Files: `xgrn_mlx/grn.py`, `xgrn_mlx/run.py`. **Do not touch any file currently being edited by Track A.**

Per-block `mx.compile` already regressed. The unexplored direction is a **single shapeless compile over the full 28-block visual forward**, pinning constants via `inputs=[...]` so MLX can fuse across block boundaries (block-N residual + block-(N+1) pre-norm + qkv weight read, etc.).

**Steps.**
1. Add a `compile_full_stack: bool = False` flag and `--compile-full-stack` CLI switch. When ON, replace the existing per-visual-pass compile with one `mx.compile(visual_forward_all_blocks, shapeless=True)` keyed on `(pt, ph, pw, compute_dtype)`. The existing `compile_visual_pass` flag must remain functional; the two are mutually exclusive.
2. Verify step-2 wall time equals step-50 wall time within 5 % to confirm no per-step retrace.
3. Strict correctness gate (debug + t2i-correct) before promotion.
4. If a shapeless compile is rejected by MLX (graph topology constraints), fall back to a **`mx.compile` with explicit shape pins** for the most common `(pt, ph, pw)` quartets used in `t2i-correct`, `t2v-short`, and `debug` — three compiles total, prewarmed by `--stable-shape-warmup`.

Acceptance: `0.25M/50` end-to-end warm repeat-5 median improves by >= 2 % vs the current 80.90 s, with `xgrn-validate` CLIP positive >= 0.93 and `xgrn-parity --full-step` exact.

---

## Track C — Late-step-only CFG and CFG-lane KV sharing

**Owner**: codex tmux `codex-c`. Files: `xgrn_mlx/run.py`, `xgrn_mlx/grn.py` (refine loop only), `xgrn_mlx/bench.py`. **Do not touch the visual-pass compile plumbing — that belongs to Track B.**

Algorithmic, not kernel. Published finding (Self-Guidance for MaskGIT, NeurIPS 2024): high CFG weight early in sampling harms quality, late-stage CFG dominates. For a 50-step `0.25M` refinement, skipping the uncond pass on the first K steps roughly halves visual-pass cost on those steps.

**Steps.**
1. Add `--cfg-start-step K` (default `0`, current behaviour) to `xgrn-run` and `xgrn-bench`. When `step < K`, run only the conditional visual forward; do not allocate uncond KV; CFG-lane KV sharing kicks in once `step >= K`.
2. Add an optional **CFG-lane KV-cache share** path: for the visual tower KV that does not depend on prompt text, cache it once per step and pass it to both lanes. Verify by parity that the shared-KV path equals the current batched path within `1e-3` bf16.
3. Sweep `K ∈ {0, 10, 20, 25, 30}` on `t2i-correct`. Report a table: `(K, end_to_end_sec, grn_refine_sec, max_rss_gb, clip_positive)`. Promote the largest `K` whose CLIP positive stays `>= 0.93`. If no K passes, keep `K=0` default and document failure in README.
4. Acceptance: same gate as Track B.

---

## Coordination

- Each tmux session keeps a scratch log at `outputs/track-{a,b,c}/log.md` with every gate-run result.
- Promotion to default (flipping any `default = True`) is a separate commit and requires a row added to README "Experiment Outcomes" with the warm repeat-5 number.
- If two tracks both want to flip defaults, the one with the larger `t2i-correct` warm repeat-5 improvement merges first; the second rebases and re-runs the gate.

## Completed (historical, kept for reference)

- Task 1 (revert `embed_visual_labels` from `mx.addmm` to matmul+add) — done.
- Task 2 (`fuse_mlp_gate_up` flag implementation) — flag landed; default flip still pending and now lives inside Track A as a prerequisite for A1's kernel sharing the gate+up output buffer.
- Task 3 (`swiglu_split` Metal kernel) — earlier attempt regressed and is opt-in only; revisited only when A1 ships and frees the per-block dispatch budget for the MLP fusion.
- Task 4 (`compile_refinement_update` random-state try/except fallback) — done.
- Task 5 (`--min-change-frac` opt-in early termination) — done; no default change until Track C settles the K sweep, since both touch the refinement loop.
