# xGRN Mac Runtime Todo

Goal: a high-performance Mac implementation for the official GRN T2I/T2V models only. Do not build a generic model runtime.

## Phase 0: Correctness Gate

- [x] Treat semantic validation failure as blocking: `outputs/orange_tabby_24steps_seed2027_optimized.png` is not an orange cat.
- [x] Add PyTorch official vs MLX numerical parity harness for one GRN block with identical weights and inputs.
- [x] Add parity checks for text prefix KV cache after block 0 and after all 28 blocks.
- [x] Add parity checks for one visual refinement forward: visual embedding, RoPE, block outputs, logits.
- [x] Add parity check for official PyTorch autoregressive one-step logits vs MLX one-step logits with identical prompt embeddings, random labels, masks, and CFG.
- [x] Verify refinement update schedule exactly: `uv run xgrn-parity --refinement-schedule --report outputs/bench/refinement-schedule-parity.json` checks `cur_pt`, shifted target `next_pt`, random mask mean, sampled-label layout, and pure-random-label reuse. Default fast mode intentionally uses target `next_pt`; `--exact-step-sync` uses sampled mask mean.
- [x] Add parity checks for `bit_label2raw_feature` and HBQ decode output: `uv run xgrn-hbq-parity --pt 1 --ph 16 --pw 16` and `--pt 2 --ph 16 --pw 16`.
- [x] Fix the first parity mismatch before continuing performance optimization: default text path is now bf16 UMT5 with fp32 prompt cache, not fp16 text-cache validation.
- [x] Define pass threshold: finite/no-NaN checks, visual logits mean abs under `1e-3`, and semantic CLIP positive score above `0.5`.
- [x] Regenerate orange-cat image only after parity checks pass: `outputs/orange_tabby_50steps_seed42_pn025M_bf16text.png`.
- [x] Add automated semantic validation CLI: `uv run xgrn-validate outputs/orange_tabby_50steps_seed42_pn025M_bf16text.png`.
- [x] Test whether more steps fix the fast preset: `outputs/orange_tabby_150steps_seed42_pn006M_bf16text.png` still failed CLIP validation, so `0.06M` remains debug-only.
- [ ] Optional quality check: run `0.25M/250` after core performance work if we need a higher-quality reference image.

## Phase 1: Mac Artifacts

- [x] Download official T2I, T2V, HBQ, and UMT5 weights.
- [x] Convert GRN transformer weights to Mac runtime fp16 shards with linear weights transposed for MLX matmul.
- [x] Convert HBQ decoder weights to a Mac runtime layout.
- [x] Convert or cache text encoder artifacts so prompt embeddings do not require repeated UMT5 loads.
- [x] Write a manifest with model shape constants, dtype, source file metadata, and key counts.

## Phase 2: Specialized Runtime

- [x] Implement GRN2b block forward in MLX or custom Metal.
- [x] Implement GRN refinement loop for `ar_discrete_GRN_bit`.
- [x] Implement prompt embedding cache lookup before loading text encoder.
- [x] Implement HBQ decode for image and streaming video output with native MLX as the default and MPS as fallback.
- [x] Keep only official GRN T2I/T2V model variants in the runtime.

## Phase 3: UI and Verification

- [x] Wire CLI and Gradio UI to the specialized runtime.
- [x] Run T2I smoke at `0.06M` with 2, 8, and 16 refinement steps.
- [x] Run T2V smoke at `0.06M` and short duration.
- [x] Record cold start, warm prompt, per-step latency, peak memory, and output paths.

## Phase 4: Performance Optimization

Status: unblocked for measured optimization. Keep `xgrn-parity --full-step` and `xgrn-validate` passing after every performance change.

- [x] Add per-stage timings to CLI and UI summaries.
- [x] Use `mx.fast.rms_norm` in the MLX GRN block.
- [x] Cache fp32 linear weights and biases lazily to avoid repeated cast work.
- [x] Add a benchmark CLI that records text cache, text encode cold/warm, GRN refine, HBQ decode, total latency, tokens/sec, peak RSS, and output validation for fixed profiles.
- [x] Run benchmark CLI smoke: `uv run xgrn-bench --profile debug --report outputs/bench/debug-report.json`.
- [x] Add original PyTorch/MPS benchmark CLI: `uv run xgrn-pytorch-bench`.
- [x] Record original PyTorch/MPS baseline in README: fp16 crashes, fp32 debug end-to-end `109.19s`.
- [x] Record current MLX speed in README: native-decoder debug cold `4.39s`, debug warm median `1.18s`.
- [x] Add benchmark `--repeat` and `--warmup` so cold-start and warm-run performance are measured separately.
- [x] Add benchmark summary aggregation: min/median/max/mean for wall time, GRN, decode, tokens/sec, RSS, and validation.
- [x] Add strict benchmark `end_to_end_sec` timing that includes generation, decode, output write, and optional validation; keep `generation_wall_sec` separately.
- [x] Split stage timings for model load, decoder load, HBQ decode compute, image materialization, output writing, and validation.
- [x] Standardize dev benchmark policy: use `--warmup --repeat 5`; report median as the main latency number, with cold run reported separately when needed.
- [x] Allow up to 250 refinement steps in the UI and add `t2i-quality-250` benchmark profile.
- [x] Add in-process prompt embedding array cache and GRN text KV cache for repeated prompt runs.
- [x] Verify warm-run cache benefit: debug repeat run dropped total time from `9.55s` to `3.25s`.
- [x] Benchmark short T2V warm repeat-5 baseline: median end-to-end `18.66s`, GRN `14.88s`, decode `3.76s`.
- [x] Benchmark remaining fixed profiles: `0.25M/50` warm repeat-5 T2I correctness median is end-to-end `80.90s`, generation `78.99s`, GRN `77.70s`, decode `1.17s`, CLIP `5/5` passed. `0.06M/150` step stress is done: end-to-end `67.04s` with CLIP, generation `60.19s`, GRN `59.87s`, decode `0.31s`, CLIP failed (`0.0010`), confirming `0.06M` is debug-only.
- [x] Benchmark current `0.25M/50` T2I fp32 correctness path after fast-stats/direct-embedding optimization: end-to-end with CLIP `109.25s`, generation `101.97s`, GRN `96.15s`, decode `5.79s`, CLIP positive `0.9733`.
- [x] Add fp32/fp16 artifact selection to CLI/UI/benchmark for performance A/B while keeping fp32 as correctness default.
- [x] Run cheap fp16 benchmark smoke: `uv run xgrn-bench --profile debug --weights-dtype fp16 --report outputs/bench/debug-fp16-report.json`.
- [x] Run `0.25M/50` fp16 semantic validation before allowing fp16 as a non-debug default: validation passed, but cold GRN refine was `225.81s`, so fp16 is not a default until warm-run and compile results are measured.
- [x] Add a stable-shape warmup path so MLX arrays, rope tables, text cache, fp32 weights, and decoder weights are materialized before measured runs: `uv run xgrn-bench --stable-shape-warmup ...` warms the same shape with two refinement steps instead of a full profile.
- [x] Add `mx.compile` experiments for the GRN block forward with fixed `pt/ph/pw`, starting with cond-only and then CFG visual passes: per-block compile regressed debug warm median from `1.75s` to `1.97s`, so it remains opt-in only.
- [x] Compile fixed-shape CFG visual pass around the full 28-block visual forward; standard debug warm repeat-5 improved to end-to-end `1.07s`, GRN `0.78s`.
- [x] Validate fixed-shape CFG visual-pass compile on the real `0.25M/50` CLIP gate: passed with warm repeat-5 median end-to-end `80.90s`, generation `78.99s`, GRN `77.70s`, decode `1.17s`, CLIP positive `0.9634`; enabled by default with `--no-compile-visual-pass` fallback.
- [x] Test compiling visual pass plus CFG logits: passed CLIP with `90.03s` end-to-end and lower RSS `8.87GB`, but slower than default; keep as opt-in `--compile-cfg-logits` low-memory mode.
- [x] Test fixed-shape compiled refinement update around categorical sampling and mask update: `--compile-refinement-update` regressed debug warm median to `1.10s`, so it remains opt-in and is not a default.
- [x] Continue remaining refinement-step fusion only where it can beat current visual-pass compile: MLX compiled update, binary sampling, precomputed pt embed, stacked CFG cache, and fused SwiGLU did not win. Remaining refinement wins require custom Metal kernels or quality-changing sparse/early-exit algorithms.
- [x] Batch conditional and unconditional CFG visual forwards with block-diagonal attention semantics, matching official CFG split behavior.
- [x] Precompute visual RoPE per refinement call and reuse visual embedding/scale token across CFG lanes.
- [x] Precompute padded batched CFG text KV once per refinement call instead of padding/concatenating per block per step.
- [x] Validate CFG batch on `0.25M/50` semantic gate: `outputs/bench/t2i-correct/latest_t2i.png` passed with positive score `0.9077`.
- [x] Compare debug end-to-end benchmark after CFG cache: warm repeat-5 median `1.75s`, GRN median `0.90s`.
- [x] Avoid repeated one-hot allocation in the hot refinement path: direct binary label embedding replaces one-hot plus `word_embed` matmul and matches the old embedding within `1e-6`.
- [x] Disable default per-step entropy and sampled-mask-mean CPU synchronization; expose `--detailed-stats` and `--exact-step-sync` for parity/debug runs.
- [x] Add GRN compute dtype A/B switch (`--compute-dtype fp32|bf16|fp16`) and benchmark bf16 debug: warm median `2.03s`, slower than fp32 `1.80s`, but lower RSS (`10.83GB` vs `13.97GB`).
- [x] Validate bf16 compute on the real `0.25M/50` correctness gate: passed CLIP with positive score `0.9618`, end-to-end `103.67s`, GRN `89.24s`, max RSS `13.73GB`; make bf16 the normal high-performance default and keep fp32 as parity/debug mode.
- [x] Validate fp16 GRN weights with bf16 compute on the real `0.25M/50` gate: passed CLIP with positive score `0.7247`, end-to-end `89.87s`, max RSS `6.31GB`; keep as `--weights-dtype fp16` low-memory mode, not speed/quality default.
- [x] Test `mx.async_eval` for the per-step mixed labels: debug warm median regressed to `2.34s`, so keep synchronous `mx.eval`.
- [x] Add optional argmax/debug sampling mode while keeping official stochastic categorical as default: `--sampling-mode argmax` benchmarked at debug warm median `1.09s`, not faster than default, so it remains a debug/quality-tradeoff mode.
- [x] Exhaust MLX-level sampling optimization: `argmax` debug warm median `1.09s`, `binary` Bernoulli debug warm median `1.127s`, and compiled update did not beat categorical; only a custom Metal sampler remains, and sampling is not the dominant cost.
- [x] Port `bit_label2raw_feature` to MLX and verify against official PyTorch: exact parity in `xgrn-hbq-parity`.
- [x] Port HBQ image decoder to native MLX.
- [x] Port HBQ video decoder to native MLX with temporal streaming cache.
- [x] Add decoder parity check against the official PyTorch/MPS decoder before enabling native MLX decoder by default: T2I-like `pt=1` mean abs `2.76e-4`, T2V-like `pt=2` mean abs `2.65e-4`; current rerun remains finite/exact for bit labels with decoder mean abs `2.76e-4` (`pt=1`) and `2.65e-4` (`pt=2`).
- [x] Make native MLX HBQ decoder the default after semantic validation: `t2i-correct` end-to-end with CLIP `100.21s`, GRN `89.52s`, decode `3.18s`, max RSS `9.87GB`, CLIP positive `0.9634`.
- [x] Verify native T2V short smoke: end-to-end `18.92s`, GRN `17.41s`, decode `1.32s`, max RSS `9.49GB`, output `outputs/bench/t2v-short/latest_t2v.mp4`.
- [x] Add GRN fp16 correctness/performance A/B against fp32 artifact after text correctness is fixed: fp16 weights + bf16 compute passed CLIP at `89.87s` / `6.31GB`, lower semantic margin than fp32 default.
- [x] Add int8 GRN linear weight-only quantization experiment with CLIP validation: passed CLIP (`0.9236`) but slowed `0.25M/50` to `105.42s` end-to-end / `96.67s` GRN, so it is not a default or preferred low-memory path.
- [x] Add int4 GRN linear quantization experiment after int8 passed semantic validation: debug warm median `1.09s`, no speed or RSS win over default, so full CLIP gate is skipped and int4 remains a non-default experiment.
- [x] Quantize/cache the text encoder only after bf16/fp32-cache semantic baseline remains reproducible: fp16 text + fp16 cache passed the loose validator but missed the strict `>=0.93` gate (`0.8493`) and slowed GRN to `86.33s`, so bf16 text + fp32 cache remains default.
- [x] Add memory-pressure controls: UMT5 is unloaded after prompt cache creation; GRN/native HBQ/MPS decoder caches expose close hooks; CLI/bench support `--release-after-run --release-text-cache`; max RSS is reported in bench rows.
- [x] Keep PyTorch/MPS decode as fallback after native MLX decoder passed parity and semantic validation; select with `--decoder-backend mps`.
- [x] CODEX Task 1: fix `--compile-refinement-update` random-state handling. Current MLX lacks `mx.compile(..., state=...)`, so the flag now falls back to the uncompiled random update instead of freezing samples; 5-step smoke showed advancing `next_pt`, and `uv run xgrn-parity --full-step` passed.
- [x] CODEX Task 2: test removing unconditional fp32 upcast in `rms_norm`; it regressed the default debug gate, so the high-performance default keeps the prior fp32 `mx.fast.rms_norm` path.
- [x] CODEX Task 3: add opt-in `--precompute-pt-embed`; debug warm repeat-3 regressed from `1.075s` to `1.092s`, so default remains on-demand pt embedding.
- [x] CODEX Task 4: add opt-in `--fuse-swiglu-metal`; fused kernel fp32 max abs diff is `2.38e-7`, but debug warm repeat-3 regressed to `1.107s`, so default keeps MLX `silu(gate) * up`.
- [x] CODEX Task 5: add opt-in adaptive early termination with `--min-change-frac`; a 0.06M/20 smoke at `0.005` did not early-stop, so it is not a default speed path.
- [x] CODEX Task 6A: add opt-in `--track-token-confidence`; smoke recorded mean confidence `0.557 -> 0.922` over 3 steps for DUS research.
- [x] CODEX Task 6B decision gate: real `0.25M/50` confidence trace does not justify packed sparse DUS now. `pct_confident_90` reaches 50% only at step 38 and 90% at step 47, while simple DUS 20/30/40 failed CLIP; do not implement sparse sequence forward until a stronger schedule or quality gate exists.
- [x] CODEX Task 7: add opt-in `--stack-cfg-cache`; debug warm repeat-3 was `1.086s`, slightly slower than default, so stacked K/V is not enabled by default.
- [x] Gate the current performance changes: `uv run python -m py_compile ...`, `uv run xgrn-parity --full-step`, current HBQ parity (`pt=1` and `pt=2`), `uv run xgrn-validate outputs/bench/t2i-correct/latest_t2i.png` positive `0.9634`, and debug benchmark CLI. Repeat these gates for future performance changes.
- [x] Software-only saturation point reached for the current MLX runtime: remaining meaningful speedups require custom Metal GEMM/sampler kernels, packed sparse sequence work with quality risk, or model-quality tradeoffs. README speed table and experiment outcomes are updated.

## Phase 5: Frontier Optimization (custom Metal + compile + algorithmic)

Driven by the M4 Pro dispatch-overhead hypothesis: ~200 Metal dispatches/step at ~6% GPU utilization at `0.25M/50`. Three independent codex tmux tracks. Each must pass the strict correctness gate in `PERFORMANCE_PLAN.md` §3 before flipping any default. See `CODEX_TASKS.md` for full briefs.

### Track A — Custom Metal kernel fusion (`mx.fast.metal_kernel`)

- [~] A0: profile capture — `mx.metal.start_capture` needs `MTL_CAPTURE_ENABLED=1` set at process spawn time; one-shot CLI capture is fiddly without Xcode in the loop. Skipped in favor of direct A1-lite implementation + bench measurement, which gave a real signal.
- [x] A1-lite: fused `apply_rope` Metal kernel. One kernel replaces the 7 elementwise dispatches in `apply_rope` (4 muls + 1 sub + 1 add + 1 stack). Behind `--fuse-rope-metal` (default OFF). Parity vs `apply_rope`: fp32 max-abs-diff `2.38e-7`, bf16 `7.56e-3`. Bench: debug warm GRN `0.772 → 0.763 s` (-1.17%), `t2i-correct` warm end-to-end `76.66 → 75.41 s` (-1.63%), RSS +236 MB, CLIP `0.9904 → 0.8985` (numerical-equivalent sample shift; loose validator 3/3, strict 0.93 missed by 0.03). Shipped opt-in.
- [x] A2: fused `(x + attn) + post-attn rms_norm` Metal kernel — multi-output, threadgroup reduction over hidden_dim=2304. Behind `--fuse-residual-norm-metal` (default OFF). Parity vs reference: residual exact, normed max-abs-diff `9.54e-7`. **Measured negative result.** `t2i-correct` warm GRN `75.56 → 75.35 s` (-0.28%, within noise), end-to-end `76.66 → 76.55 s` (-0.15%), **RSS +7.09% (+684 MB)** because the two output buffers cannot be freed in-place the way MLX's compile manages residual reuse. 28 dispatches saved per step is too few to outweigh the buffer-allocation cost. Flag kept as a template for A3 / A1-full multi-output kernels but **do not enable**.
- [~] A1-full: fused `rms_norm + q_proj` Metal kernel.
    - [x] POC scaffold (`fused_norm_qproj` in `xgrn_mlx/grn.py`) with naive scalar matmul: parity perfect (fp32 max-abs-diff 2.86e-6), 8× slower than MLX, not wireable. Proves the fusion structure.
    - [x] **Simdgroup matmul foundation landed** in `xgrn_mlx/simdgroup_matmul.py`, iterated through 10 layouts to convergence:
        - Winner: 2×2 simdgroup per threadgroup + 2×4 register tile per simdgroup (8 accumulators)
        - **fp32: 1.045× of MLX matmul, byte-exact parity** (max_abs_diff 0.0)
        - **bf16: 0.975× of MLX bf16 matmul — i.e. ~2.5 % FASTER than MLX**, within bf16 precision
        - Output-stationary register tiling was the winning structural move (1-acc 2.13× → 8-acc 1.05×)
        - 4×3 (12 acc) and 4×4 (16 acc) regressed badly (5.65×, 6.32×) — register spill threshold sits between 8 and 12 accumulators
        - Threadgroup-memory tile caching regressed (3.32×) — barrier cost > L2 cache dedup savings
        - `simd_async_copy` requires Metal 3's `simdgroup_event` header (not in MLX toolchain) — deferred until toolchain catches up
    - [~] M3+M5 attempted as one change: `--fuse-qkv-metal` flag routes the 3 qkv projection matmuls through `simdgroup_matmul_bf16`. Wired but **measured negative on t2i-correct**: GRN `75.56 → 79.57 s` (+5.31 %), end-to-end `76.66 → 83.45 s` (+8.86 %), **decode `1.06 → 1.99 s` (+88 %, indirect — decoder doesn't use qkv)**, CLIP `0.9904 → 0.9206`. The decode jump despite untouched decoder code points to the same integration tax A2 hit: opaque Metal kernels inside the `mx.compile`'d visual pass break fusion / scheduling MLX's native matmul gets. Flag stays opt-in only.
    - [~] M3 alt (`--fuse-qkv-concat`, MLX-only path): one MLX matmul against the concatenated `[q|k|v]` weight per block, no custom Metal kernel. Tests whether the M3 regression was a Metal-kernel issue or a fusion-shape issue. Result: **also regresses on t2i-correct** (+4.75 % GRN, +5.32 % wall, +10.63 % decode, CLIP 0.9904 → 0.8999). Confirms the mx.compile'd visual pass already optimizes the 3 separate qkv matmuls better than any manual fusion. Flag landed for evidence; do not enable.
    - [~] MLX C++ extension (`xgrn_ext/`) **landed**. Build pipeline works (CMake + nanobind + `NB_DOMAIN mlx`, no Metal Toolchain required, runtime MSL compile via `mx::fast::metal_kernel` from C++). Isolated microbench bf16 ~1.01× of MLX (within noise; was 0.97× before kernel cache stabilized). `--fuse-qkv-ext` integration result: t2i-correct +3.77 % GRN, +4.09 % wall, +15.23 % decode — best of all QKV-fusion attempts but still net negative. The compile-graph treats `mx.fast.metal_kernel` as opaque whether dispatched from Python or C++.
    - [x] Custom Primitive subclass landed end-to-end after `xcodebuild -downloadComponent MetalToolchain` (~688 MB, no sudo needed on this system). `SimdgroupMatmul : public mlx::core::Primitive`, axpby-style eval_gpu, precompiled `xgrn_ext.metallib`. Byte-exact parity vs raw kernel + microbench at parity with MLX matmul (~1.00×). **`mx::fast::Custom` cannot be inherited from extensions** — libmlx.dylib doesn't export its vtable / vjp / vmap symbols; we inherit `Primitive` directly like the upstream `axpby` example.
    - [x] `--fuse-qkv-prim` flag wired through. Integration result on t2i-correct: GRN +7.30 %, end-to-end +8.74 %, decode +45.66 %, CLIP 0.9904 → 0.9206. **Worse than `--fuse-qkv-ext`** (+3.77 % GRN) because axpby-style per-call overhead (allocator::malloc + set_input_array × 3 + set_bytes × 3 + dispatch_threads) accumulates across 1400 qkv calls/generation. MLX's internal CustomKernel infrastructure presumably has buffer-pool reuse + encoder batching we can't replicate via the public extension API. Named-primitive status doesn't unlock additional mx.compile fusion in practice — compile's fusion rules are op-specific, not generic. Flag stays opt-in for parity with the other QKV experiments; do not enable.
    - **Verdict for Track A1-full:** every available framework integration path tried (raw `mx.fast.metal_kernel` from Python, raw from C++, Python-only MLX concat, axpby-style Custom Primitive). All regress at t2i-correct gate by 4-9 % GRN. The bottleneck is the integration tax custom kernels pay inside the visual_pass compile, not the kernel quality (our kernels match or beat MLX matmul in isolation). Net wins on this stack would need either (a) a deeper compile-pass integration MLX doesn't expose to extensions, (b) a model-architecture-level change (Track D step distillation), or (c) hardware/MLX-version upgrade.
- [~] A3: fused sampling + mask update kernel with `atomic_outputs=True`. **Deferred.** Per-step dispatch saving (~5/step × 50 steps = 250/generation) is too small relative to A1-lite's 16,800/generation. Dispatch-count theory predicts -0.05% to -0.15% wall — deep in noise — and the A2 RSS regression pattern would likely repeat. Not worth the implementation cost given the saturation we've already reached on non-matmul fusion.
- [ ] Promote any A track default ON when aggregate `t2i-correct` warm repeat-5 median drops by >= 5 % and the gate is **the recalibrated multi-prompt loose gate**, not the prompt-specific 0.93 strict gate that turned out to be a calibration artifact (see Track C findings).

### Track B — Whole-stack `mx.compile(shapeless=True)`

- [ ] Add `--compile-full-stack` flag (mutually exclusive with `--compile-visual-pass`). Single shapeless compile over all 28 blocks, keyed on `(pt, ph, pw, compute_dtype)`.
- [ ] Verify step-2 vs step-50 wall time within 5 % (no per-step retrace).
- [ ] Fallback: if shapeless rejected, ship explicit shape pins for `debug`, `t2i-correct`, `t2v-short` quartets warmed by `--stable-shape-warmup`.
- [ ] Promote default ON when `t2i-correct` warm median improves by >= 2 % with gates passing.

### Track C — Late-step-only CFG and CFG-lane KV sharing

- [x] Add `--cfg-start-step K` (default 0). When `step < K`, run only the conditional visual forward (existing `visual_forward_embedded` with `cond_cache`). K=0 is bit-identical to today (verified by SHA-256 PNG match).
- [x] Sweep `K ∈ {0, 5, 10, 15, 17, 20, 25}` on `t2i-correct`. K=15 saves 13.8% wall on the standard prompt (76.66 → 66.10 s end-to-end) with CLIP 0.9635. K=10 passes razor-thin (0.9332). K=17/20/25 fail strict 0.93 gate.
- [x] Decision: ship `--cfg-start-step K` as opt-in flag, default K=0 (no default change). Quality is non-monotonic in K → prompt/seed sensitivity makes a default unsafe without a multi-prompt validation set.
- [ ] CFG-lane KV-cache share for visual-tower KV that does not depend on prompt text — still proposed, not implemented.
- [~] Multi-prompt K stability sweep (5–10 prompts, K ∈ {10, 12, 15}). Started 2026-05-15; stopped after orange_tabby (only one prompt completed) because running a 35-min foreground GPU sweep alongside the live demo on port 7860 violates the "no concurrent GPU tests" rule. Two findings worth keeping: (a) the 0.93 strict gate does NOT transfer across CLIP negative sets — same K=0 image scored 0.685 against harder distractors vs 0.9904 against the original easy ones; (b) K=10's image's top CLIP label flips to "a blurry distorted image" against harder distractors, exposing quality cost the original gate hid. Required before any default promotion: rerun with prompt-specific positives + harder shared negatives, recalibrated loose gate.
- [ ] Compile the cond-only `visual_forward_embedded` for B=1 fixed-shape too. Would tighten the K>0 savings curve and shrink the +0.8 GB RSS overhead during steps < K.

### Track D — Step distillation (future, training-side)

- [ ] Decide whether to fund a self-distillation training run (DiMO / CDLM style) to compress 50 steps → 8–12 steps for discrete masked refinement. Out of scope for current runtime work; tracked as the highest remaining ceiling.
