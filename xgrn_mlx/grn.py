from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from .constants import GRN2B, GRN2BConfig
from .rope import apply_rope, text_rope, visual_rope
from .schedule import refinement_target_pt


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), weight.astype(mx.float32), eps)


_HAS_METAL_KERNEL = hasattr(mx.fast, "metal_kernel")
_SWIGLU_KERNEL = None
_FUSED_ROPE_KERNEL = None
_FUSED_RESIDUAL_RMSNORM_KERNEL = None


def fused_rope(x: mx.array, rope: mx.array) -> mx.array:
    """One-kernel rotary embedding for x [B, H, L, D] and rope [2, L, D/2].

    Replaces 7 elementwise dispatches in apply_rope with a single Metal
    kernel. Output layout matches apply_rope exactly: interleaved pairs
    `(x[..., 2i], x[..., 2i+1]) -> (a, b)` where
    `a = x[..., 2i] * cos - x[..., 2i+1] * sin` and
    `b = x[..., 2i] * sin + x[..., 2i+1] * cos`.
    """
    if not _HAS_METAL_KERNEL:
        from .rope import apply_rope
        return apply_rope(x, rope)
    global _FUSED_ROPE_KERNEL
    if _FUSED_ROPE_KERNEL is None:
        source = """
            uint idx = thread_position_in_grid.x;
            uint total = N;
            if (idx >= total) return;

            uint d = idx % D;
            uint l = (idx / D) % L;
            uint pair_idx = d >> 1;
            bool is_odd = (d & 1u) != 0u;
            uint half_d = D >> 1;
            uint rope_cos_off = l * half_d + pair_idx;
            uint rope_sin_off = L * half_d + l * half_d + pair_idx;

            float cos_v = static_cast<float>(rope[rope_cos_off]);
            float sin_v = static_cast<float>(rope[rope_sin_off]);

            uint pair_base = idx & ~1u;
            float x_even = static_cast<float>(x[pair_base]);
            float x_odd = static_cast<float>(x[pair_base | 1u]);

            float result = is_odd
                ? (x_even * sin_v + x_odd * cos_v)
                : (x_even * cos_v - x_odd * sin_v);
            out[idx] = static_cast<T>(result);
        """
        _FUSED_ROPE_KERNEL = mx.fast.metal_kernel(
            name="xgrn_fused_rope",
            input_names=["x", "rope"],
            output_names=["out"],
            source=source,
            header="#include <metal_math>\n",
        )
    *_, L, D = x.shape
    out = _FUSED_ROPE_KERNEL(
        inputs=[x, rope],
        template=[("T", x.dtype), ("L", L), ("D", D), ("N", x.size)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1),
        threadgroup=(min(x.size, 256), 1, 1),
        verbose=False,
    )
    return out[0] if isinstance(out, (list, tuple)) else out


def fused_residual_rmsnorm(
    x: mx.array,
    attn: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """One kernel: (residual, normed) = (x+attn, rms_norm(x+attn, weight)).

    Replaces 2 dispatches (the elementwise residual add and the
    `mx.fast.rms_norm` call) with one Metal kernel. Layout: one
    threadgroup per token, threads cooperatively partial-sum the
    squared residual into threadgroup memory, tree-reduce to a single
    inv-sigma, then second pass writes both outputs. fp32 accumulation
    independent of input dtype.

    Inputs:
      x      : [..., H] residual to add to
      attn   : [..., H] attention output, same shape as x
      weight : [H]       rms-norm scale
    Outputs:
      residual : [..., H] = x + attn, dtype matches x
      normed   : [..., H] = (x + attn) * rsqrt(mean((x+attn)^2) + eps) * weight
                 dtype matches x
    """
    if not _HAS_METAL_KERNEL:
        residual = x + attn
        return residual, rms_norm(residual, weight, eps)
    global _FUSED_RESIDUAL_RMSNORM_KERNEL
    if _FUSED_RESIDUAL_RMSNORM_KERNEL is None:
        # EPS is inlined via Python format because mx.fast.metal_kernel
        # template params accept only dtype / int / bool, not float.
        source_template = """
            float kEpsilon = {eps_literal}f;

            uint tid = thread_position_in_threadgroup.x;
            uint token_id = threadgroup_position_in_grid.x;
            uint base = token_id * H;

            threadgroup float ssq_shared[TPG];

            float ssq = 0.0f;
            for (uint i = tid; i < H; i += TPG) {{
                float r = static_cast<float>(x[base + i]) + static_cast<float>(attn[base + i]);
                ssq += r * r;
            }}
            ssq_shared[tid] = ssq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = TPG >> 1; stride > 0; stride >>= 1) {{
                if (tid < stride) {{
                    ssq_shared[tid] += ssq_shared[tid + stride];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            float mean_sq = ssq_shared[0] / static_cast<float>(H);
            float inv_sigma = metal::rsqrt(mean_sq + kEpsilon);

            for (uint i = tid; i < H; i += TPG) {{
                float r = static_cast<float>(x[base + i]) + static_cast<float>(attn[base + i]);
                residual[base + i] = static_cast<T>(r);
                float n = r * inv_sigma * static_cast<float>(weight[i]);
                normed[base + i] = static_cast<T>(n);
            }}
        """
        source = source_template.format(eps_literal=repr(float(eps)))
        _FUSED_RESIDUAL_RMSNORM_KERNEL = mx.fast.metal_kernel(
            name="xgrn_fused_residual_rmsnorm",
            input_names=["x", "attn", "weight"],
            output_names=["residual", "normed"],
            source=source,
            header="#include <metal_math>\n",
        )
    H = x.shape[-1]
    num_tokens = x.size // H
    TPG = 256
    outs = _FUSED_RESIDUAL_RMSNORM_KERNEL(
        inputs=[x, attn, weight],
        template=[("T", x.dtype), ("H", H), ("TPG", TPG)],
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
        grid=(num_tokens * TPG, 1, 1),
        threadgroup=(TPG, 1, 1),
        verbose=False,
    )
    if isinstance(outs, (list, tuple)):
        return outs[0], outs[1]
    return outs.residual, outs.normed


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    if not _HAS_METAL_KERNEL:
        return silu(gate) * up
    global _SWIGLU_KERNEL
    if _SWIGLU_KERNEL is None:
        source = """
            uint idx = thread_position_in_grid.x;
            float g = static_cast<float>(gate[idx]);
            float u = static_cast<float>(up[idx]);
            float sig = 1.0f / (1.0f + metal::exp(-g));
            out[idx] = static_cast<T>(g * sig * u);
        """
        _SWIGLU_KERNEL = mx.fast.metal_kernel(
            name="xgrn_swiglu_fused",
            input_names=["gate", "up"],
            output_names=["out"],
            source=source,
            header="#include <metal_math>\n",
        )
    out = _SWIGLU_KERNEL(
        inputs=[gate, up],
        template=[("T", gate.dtype)],
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
        grid=(gate.size, 1, 1),
        threadgroup=(min(gate.size, 256), 1, 1),
        verbose=False,
    )
    return out[0] if isinstance(out, (list, tuple)) else out


def one_hot(labels: mx.array, classes: int) -> mx.array:
    return mx.eye(classes, dtype=mx.float16)[labels.astype(mx.int32)]


def timestep_embedding(t: float, dim: int = 256) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-np.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
    args = mx.array([t], dtype=mx.float32)[:, None] * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb.reshape(1, 1, dim).astype(mx.float32)


SAMPLING_MODES = {"categorical", "binary", "argmax"}
MASK_SCHEDULES = {"random", "dus"}


@dataclass
class KVCache:
    k: list[mx.array]
    v: list[mx.array]


@dataclass
class CFGKVCache:
    k: list[mx.array]
    v: list[mx.array]
    mask: mx.array
    k_stacked: mx.array | None = None
    v_stacked: mx.array | None = None


class GRN2BMLX:
    def __init__(
        self,
        weights_path: str | Path,
        config: GRN2BConfig = GRN2B,
        compute_dtype: str = "fp32",
        compile_blocks: bool = False,
        compile_visual_pass: bool = False,
        compile_cfg_logits: bool = False,
        compile_refinement_update: bool = False,
        linear_quantization: str = "none",
        fuse_mlp_gate_up: bool = False,
        fuse_swiglu_metal: bool = False,
        fuse_rope_metal: bool = False,
        fuse_residual_norm_metal: bool = False,
        stack_cfg_cache: bool = False,
    ):
        self.config = config
        if compute_dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"unsupported compute dtype {compute_dtype}")
        if linear_quantization not in {"none", "int8", "int4"}:
            raise ValueError(f"unsupported linear quantization {linear_quantization}")
        self.compute_dtype = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}[compute_dtype]
        self.linear_quantization = linear_quantization
        self.compile_blocks = compile_blocks
        self.compile_visual_pass = compile_visual_pass
        self.compile_cfg_logits = compile_cfg_logits
        self.compile_refinement_update = compile_refinement_update
        self.w = mx.load(str(weights_path))
        self.fuse_mlp_gate_up = fuse_mlp_gate_up
        self.fuse_swiglu_metal = fuse_swiglu_metal
        self.fuse_rope_metal = fuse_rope_metal
        self.fuse_residual_norm_metal = fuse_residual_norm_metal
        self.stack_cfg_cache = stack_cfg_cache
        self.w32: dict[str, mx.array] = {}
        self.w_compute: dict[str, mx.array] = {}
        self.w_quant: dict[str, tuple[mx.array, mx.array, mx.array]] = {}
        self.w_mlp_gate_up: dict[int, tuple[mx.array, mx.array | None]] = {}
        self.text_kv_cache: dict[str, KVCache] = {}
        self.compiled_text_blocks: dict[int, Callable] = {}
        self.compiled_visual_blocks: dict[int, Callable] = {}
        self.compiled_cfg_visual_pass: Callable | None = None
        self.compiled_cfg_logits_passes: dict[tuple[int, float, float], Callable] = {}
        self.compiled_refinement_updates: dict[tuple[int, int, int, int, int], Callable | None] = {}
        self.dus_ranks: dict[tuple[int, int, int], mx.array] = {}
        self._word_embed_label_base: mx.array | None = None
        self._word_embed_label_delta: mx.array | None = None

    def fp32_weight(self, name: str) -> mx.array:
        if name not in self.w32:
            self.w32[name] = self.w[name].astype(mx.float32)
            mx.eval(self.w32[name])
        return self.w32[name]

    def compute_weight(self, name: str) -> mx.array:
        if self.compute_dtype == mx.float32:
            return self.fp32_weight(name)
        if name not in self.w_compute:
            self.w_compute[name] = self.w[name].astype(self.compute_dtype)
            mx.eval(self.w_compute[name])
        return self.w_compute[name]

    def quantized_weight(self, name: str) -> tuple[mx.array, mx.array, mx.array]:
        if name not in self.w_quant:
            # Stored 2D weights are already transposed for x @ W. MLX's fast
            # quantized path is x @ Wq.T, so quantize the untransposed view.
            bits = int(self.linear_quantization.removeprefix("int"))
            weight = mx.transpose(self.compute_weight(name))
            qweight, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
            mx.eval(qweight, scales, biases)
            self.w_quant[name] = (qweight, scales, biases)
        return self.w_quant[name]

    def mlp_gate_up_weight(self, block: int) -> tuple[mx.array, mx.array | None]:
        if block not in self.w_mlp_gate_up:
            gate_prefix = self.key(block, "mlp.gate_proj")
            up_prefix = self.key(block, "mlp.up_proj")
            gate = self.compute_weight(f"{gate_prefix}.weight")
            up = self.compute_weight(f"{up_prefix}.weight")
            weight = mx.concatenate([gate, up], axis=-1)
            bias = None
            gate_bias_name = f"{gate_prefix}.bias"
            up_bias_name = f"{up_prefix}.bias"
            if gate_bias_name in self.w or up_bias_name in self.w:
                gate_bias = self.compute_weight(gate_bias_name) if gate_bias_name in self.w else mx.zeros((gate.shape[-1],), dtype=self.compute_dtype)
                up_bias = self.compute_weight(up_bias_name) if up_bias_name in self.w else mx.zeros((up.shape[-1],), dtype=self.compute_dtype)
                bias = mx.concatenate([gate_bias, up_bias], axis=-1)
            if bias is None:
                mx.eval(weight)
            else:
                mx.eval(weight, bias)
            self.w_mlp_gate_up[block] = (weight, bias)
        return self.w_mlp_gate_up[block]

    def key(self, block: int, suffix: str) -> str:
        chunk = block // self.config.blocks_per_chunk
        inner = block % self.config.blocks_per_chunk
        return f"block_chunks.{chunk}.module.{inner}.{suffix}"

    def linear(self, x: mx.array, prefix: str) -> mx.array:
        weight_name = f"{prefix}.weight"
        if self.linear_quantization == "int8":
            qweight, scales, biases = self.quantized_weight(weight_name)
            y = mx.quantized_matmul(
                x.astype(self.compute_dtype),
                qweight,
                scales,
                biases,
                transpose=True,
                group_size=64,
                bits=int(self.linear_quantization.removeprefix("int")),
            )
        elif self.compute_dtype == mx.float32:
            y = x.astype(mx.float32) @ self.fp32_weight(f"{prefix}.weight")
        else:
            y = x.astype(self.compute_dtype) @ self.compute_weight(weight_name)
        bias_name = f"{prefix}.bias"
        if bias_name in self.w:
            bias = self.compute_weight(bias_name)
            y = y + bias
        return y.astype(mx.float32) if self.compute_dtype != mx.float32 else y

    def block_linear(self, x: mx.array, block: int, suffix: str) -> mx.array:
        return self.linear(x, self.key(block, suffix))

    def pt_embed(self, pt: float) -> mx.array:
        x = timestep_embedding(pt)
        x = self.linear(x, "pt_embedder.mlp.0")
        x = silu(x)
        return self.linear(x, "pt_embedder.mlp.2")

    def project_text(self, text: mx.array) -> mx.array:
        if text.ndim == 2:
            text = text[None]
        return self.linear(text, "text_proj")

    def embed_visual_codes(self, onehot_bcthw: mx.array) -> mx.array:
        b, c, pt, ph, pw = onehot_bcthw.shape
        x = onehot_bcthw.reshape(b, c, pt * ph * pw)
        x = mx.transpose(x, (0, 2, 1))
        return self.linear(x, "word_embed")

    def _word_embed_label_weights(self) -> tuple[mx.array, mx.array]:
        if self._word_embed_label_base is None or self._word_embed_label_delta is None:
            weight = self.fp32_weight("word_embed.weight")
            zeros = weight[0::2]
            ones = weight[1::2]
            base = mx.sum(zeros, axis=0)
            if "word_embed.bias" in self.w:
                base = base + self.fp32_weight("word_embed.bias")
            self._word_embed_label_base = base
            self._word_embed_label_delta = ones - zeros
            mx.eval(self._word_embed_label_base, self._word_embed_label_delta)
        return self._word_embed_label_base, self._word_embed_label_delta

    def embed_visual_labels(self, labels: mx.array) -> mx.array:
        b, d, pt, ph, pw = labels.shape
        base, delta = self._word_embed_label_weights()
        labels_by_token = mx.transpose(labels, (0, 2, 3, 4, 1)).reshape(b, pt * ph * pw, d)
        return mx.addmm(base, labels_by_token.astype(mx.float32), delta)

    def qkv(
        self,
        x: mx.array,
        block: int,
        rope: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        cfg = self.config
        b, l, _ = x.shape
        q = self.block_linear(x, block, "attn.q_proj").reshape(b, l, cfg.num_heads, cfg.head_dim)
        k = self.block_linear(x, block, "attn.k_proj").reshape(b, l, cfg.num_key_value_heads, cfg.head_dim)
        v = self.block_linear(x, block, "attn.v_proj").reshape(b, l, cfg.num_key_value_heads, cfg.head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        q = rms_norm(q, self.w[self.key(block, "attn.q_norm.weight")])
        k = rms_norm(k, self.w[self.key(block, "attn.k_norm.weight")])
        if self.fuse_rope_metal:
            q = fused_rope(q, rope)
            k = fused_rope(k, rope)
        else:
            q = apply_rope(q, rope)
            k = apply_rope(k, rope)
        return q, k, v

    def attention(
        self,
        x: mx.array,
        block: int,
        rope: mx.array,
        prefix_k: mx.array | None = None,
        prefix_v: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        cfg = self.config
        b, l, _ = x.shape
        q, k, v = self.qkv(x, block, rope)
        cur_k, cur_v = k, v
        if prefix_k is not None:
            k = mx.concatenate([prefix_k, k], axis=2)
            v = mx.concatenate([prefix_v, v], axis=2)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=cfg.head_dim**-0.5, mask=mask)
        y = mx.transpose(y, (0, 2, 1, 3)).reshape(b, l, cfg.embed_dim)
        return self.block_linear(y, block, "attn.o_proj"), cur_k, cur_v

    def mlp(self, x: mx.array, block: int) -> mx.array:
        if self.fuse_mlp_gate_up and self.linear_quantization == "none":
            weight, bias = self.mlp_gate_up_weight(block)
            gate_up = x.astype(self.compute_dtype) @ weight
            if bias is not None:
                gate_up = gate_up + bias
            if self.compute_dtype != mx.float32:
                gate_up = gate_up.astype(mx.float32)
            hidden = self.config.mlp_hidden_dim
            gate = gate_up[..., :hidden]
            up = gate_up[..., hidden:]
        else:
            gate = self.block_linear(x, block, "mlp.gate_proj")
            up = self.block_linear(x, block, "mlp.up_proj")
        hidden = swiglu(gate, up) if self.fuse_swiglu_metal else silu(gate) * up
        return self.block_linear(hidden, block, "mlp.down_proj")

    def block(
        self,
        x: mx.array,
        block: int,
        rope: mx.array,
        prefix_k: mx.array | None = None,
        prefix_v: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        h = rms_norm(x, self.w[self.key(block, "input_layernorm.weight")])
        attn, cur_k, cur_v = self.attention(h, block, rope, prefix_k, prefix_v, mask)
        if self.fuse_residual_norm_metal:
            # The kernel does fp32 accumulation internally and casts the
            # weight inline via static_cast<float>, so we can pass the raw
            # (possibly fp16) weight directly without an extra astype dispatch.
            post_w = self.w[self.key(block, "post_attention_layernorm.weight")]
            x, h = fused_residual_rmsnorm(x, attn, post_w)
        else:
            x = x + attn
            h = rms_norm(x, self.w[self.key(block, "post_attention_layernorm.weight")])
        x = x + self.mlp(h, block)
        return x, cur_k, cur_v

    def block_maybe_compiled(
        self,
        x: mx.array,
        block: int,
        rope: mx.array,
        prefix_k: mx.array | None = None,
        prefix_v: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if not self.compile_blocks:
            return self.block(x, block, rope, prefix_k, prefix_v, mask)
        if prefix_k is None and mask is None:
            if block not in self.compiled_text_blocks:
                self.compiled_text_blocks[block] = mx.compile(lambda x_, rope_, b=block: self.block(x_, b, rope_))
            return self.compiled_text_blocks[block](x, rope)
        if prefix_k is not None and prefix_v is not None and mask is not None:
            if block not in self.compiled_visual_blocks:
                self.compiled_visual_blocks[block] = mx.compile(
                    lambda x_, rope_, pk_, pv_, mask_, b=block: self.block(x_, b, rope_, pk_, pv_, mask_)
                )
            return self.compiled_visual_blocks[block](x, rope, prefix_k, prefix_v, mask)
        return self.block(x, block, rope, prefix_k, prefix_v, mask)

    def encode_text_cache(self, text: mx.array) -> KVCache:
        x = self.project_text(text)
        rope = text_rope(x.shape[1])
        keys: list[mx.array] = []
        vals: list[mx.array] = []
        for block in range(self.config.depth):
            x, k, v = self.block_maybe_compiled(x, block, rope)
            keys.append(k)
            vals.append(v)
        mx.eval(x, *keys, *vals)
        return KVCache(keys, vals)

    def encode_text_cache_cached(self, text: mx.array, cache_key: str | None = None) -> KVCache:
        if cache_key is None:
            return self.encode_text_cache(text)
        if cache_key not in self.text_kv_cache:
            self.text_kv_cache[cache_key] = self.encode_text_cache(text)
        return self.text_kv_cache[cache_key]

    def visual_forward(self, visual: mx.array, pt_token: float, pt: int, ph: int, pw: int, mapped_h_div_w: float, cache: KVCache) -> mx.array:
        x = self.embed_visual_codes(visual)
        scale = self.pt_embed(pt_token)
        x = mx.concatenate([x, scale], axis=1)
        rope = mx.concatenate([visual_rope(pt, ph, pw, mapped_h_div_w), text_rope(1, offset=512)], axis=1)
        return self.visual_forward_embedded(x, rope, cache)

    def visual_forward_embedded(self, x: mx.array, rope: mx.array, cache: KVCache) -> mx.array:
        for block in range(self.config.depth):
            x, _, _ = self.block_maybe_compiled(x, block, rope, cache.k[block], cache.v[block])
        return x

    def make_cfg_cache(self, cond_cache: KVCache, uncond_cache: KVCache, visual_len: int) -> CFGKVCache:
        cond_len = cond_cache.k[0].shape[2]
        uncond_len = uncond_cache.k[0].shape[2]
        max_text = max(cond_len, uncond_len)
        total_k = max_text + visual_len
        cond_valid = np.concatenate([np.arange(max_text) < cond_len, np.ones(visual_len, dtype=bool)])
        uncond_valid = np.concatenate([np.arange(max_text) < uncond_len, np.ones(visual_len, dtype=bool)])
        mask = mx.array(np.stack([cond_valid, uncond_valid]).reshape(2, 1, 1, total_k))
        keys: list[mx.array] = []
        vals: list[mx.array] = []
        for block in range(self.config.depth):
            ck = cond_cache.k[block]
            cv = cond_cache.v[block]
            uk = uncond_cache.k[block]
            uv = uncond_cache.v[block]
            if ck.shape[2] < max_text:
                ck = mx.pad(ck, [(0, 0), (0, 0), (0, max_text - ck.shape[2]), (0, 0)])
                cv = mx.pad(cv, [(0, 0), (0, 0), (0, max_text - cv.shape[2]), (0, 0)])
            if uk.shape[2] < max_text:
                uk = mx.pad(uk, [(0, 0), (0, 0), (0, max_text - uk.shape[2]), (0, 0)])
                uv = mx.pad(uv, [(0, 0), (0, 0), (0, max_text - uv.shape[2]), (0, 0)])
            keys.append(mx.concatenate([ck, uk], axis=0))
            vals.append(mx.concatenate([cv, uv], axis=0))
        k_stacked = mx.stack(keys, axis=0) if self.stack_cfg_cache else None
        v_stacked = mx.stack(vals, axis=0) if self.stack_cfg_cache else None
        if k_stacked is not None and v_stacked is not None:
            mx.eval(mask, k_stacked, v_stacked)
        else:
            mx.eval(mask, *keys, *vals)
        return CFGKVCache(keys, vals, mask, k_stacked, v_stacked)

    def visual_forward_cfg_batched(self, x: mx.array, rope: mx.array, cache: CFGKVCache) -> mx.array:
        if self.compile_visual_pass:
            return self.visual_forward_cfg_batched_compiled(x, rope, cache)
        x = mx.concatenate([x, x], axis=0)
        for block in range(self.config.depth):
            x, _, _ = self.block_maybe_compiled(x, block, rope, cache.k[block], cache.v[block], cache.mask)
        return x

    def visual_forward_cfg_batched_compiled(self, x: mx.array, rope: mx.array, cache: CFGKVCache) -> mx.array:
        if self.stack_cfg_cache and cache.k_stacked is not None and cache.v_stacked is not None:
            return self.visual_forward_cfg_batched_compiled_stacked(x, rope, cache)
        if self.compiled_cfg_visual_pass is None:
            depth = self.config.depth

            def cfg_visual_pass(x_: mx.array, rope_: mx.array, mask_: mx.array, *kv: mx.array) -> mx.array:
                keys = kv[:depth]
                vals = kv[depth:]
                y = mx.concatenate([x_, x_], axis=0)
                for block in range(depth):
                    y, _, _ = self.block(y, block, rope_, keys[block], vals[block], mask_)
                return y

            self.compiled_cfg_visual_pass = mx.compile(cfg_visual_pass)
        return self.compiled_cfg_visual_pass(x, rope, cache.mask, *(cache.k + cache.v))

    def visual_forward_cfg_batched_compiled_stacked(self, x: mx.array, rope: mx.array, cache: CFGKVCache) -> mx.array:
        if self.compiled_cfg_visual_pass is None:
            depth = self.config.depth

            def cfg_visual_pass(x_: mx.array, rope_: mx.array, mask_: mx.array, k_stacked_: mx.array, v_stacked_: mx.array) -> mx.array:
                y = mx.concatenate([x_, x_], axis=0)
                for block in range(depth):
                    y, _, _ = self.block(y, block, rope_, k_stacked_[block], v_stacked_[block], mask_)
                return y

            self.compiled_cfg_visual_pass = mx.compile(cfg_visual_pass)
        return self.compiled_cfg_visual_pass(x, rope, cache.mask, cache.k_stacked, cache.v_stacked)

    def cfg_logits_compiled(
        self,
        x: mx.array,
        rope: mx.array,
        cache: CFGKVCache,
        token_count: int,
        guidance: float,
        temperature: float,
    ) -> mx.array:
        compile_key = (int(token_count), float(guidance), float(temperature))
        if compile_key not in self.compiled_cfg_logits_passes:
            depth = self.config.depth
            temp = float(temperature)
            cfg = float(guidance)

            def cfg_logits_pass(x_: mx.array, rope_: mx.array, mask_: mx.array, *kv: mx.array) -> mx.array:
                keys = kv[:depth]
                vals = kv[depth:]
                y = mx.concatenate([x_, x_], axis=0)
                for block in range(depth):
                    y, _, _ = self.block(y, block, rope_, keys[block], vals[block], mask_)
                cfg_logits = self.logits(y, token_count)
                cond_logits = cfg_logits[:1]
                uncond_logits = cfg_logits[1:2]
                return (uncond_logits + cfg * (cond_logits - uncond_logits)) / temp

            self.compiled_cfg_logits_passes[compile_key] = mx.compile(cfg_logits_pass)
        return self.compiled_cfg_logits_passes[compile_key](x, rope, cache.mask, *(cache.k + cache.v))

    def logits(self, x: mx.array, token_count: int) -> mx.array:
        x = rms_norm(x, self.w["head.norm.weight"])
        x = self.linear(x, "head.proj")
        x = x[:, :token_count]
        return x.reshape(x.shape[0], token_count, -1, self.config.detail_num_lvl)

    def labels_to_onehot(self, labels: mx.array) -> mx.array:
        b, d, pt, ph, pw = labels.shape
        oh = one_hot(labels, self.config.detail_num_lvl)
        oh = mx.transpose(oh, (0, 1, 5, 2, 3, 4))
        return oh.reshape(b, d * self.config.detail_num_lvl, pt, ph, pw).astype(mx.float16)

    def bit_labels_to_raw(self, labels: mx.array) -> mx.array:
        b, hd, pt, ph, pw = labels.shape
        d = hd // self.config.hbq_round
        bits = labels.reshape(b, self.config.hbq_round, d, pt, ph, pw).astype(mx.int32)
        raw = mx.zeros((b, d, pt, ph, pw), dtype=mx.float16)
        for i in range(self.config.hbq_round):
            interval = np.float16((1 / 2) ** (i + 1))
            raw = raw + mx.where(bits[:, i] == 1, interval, -interval).astype(mx.float16)
        return raw

    def dus_rank(self, pt: int, ph: int, pw: int) -> mx.array:
        key = (int(pt), int(ph), int(pw))
        if key not in self.dus_ranks:
            coords = np.indices((pt, ph, pw), dtype=np.uint64)
            # Deterministic dilated order: the multiplicative hash spreads early
            # prefixes across time/height/width instead of revealing adjacent runs.
            hashed = (
                coords[0] * np.uint64(0x9E3779B185EBCA87)
                ^ coords[1] * np.uint64(0xC2B2AE3D27D4EB4F)
                ^ coords[2] * np.uint64(0x165667B19E3779F9)
            )
            order = np.argsort(hashed.reshape(-1), kind="stable")
            ranks = np.empty(order.shape[0], dtype=np.float32)
            ranks[order] = (np.arange(order.shape[0], dtype=np.float32) + 0.5) / order.shape[0]
            self.dus_ranks[key] = mx.array(ranks.reshape(1, 1, pt, ph, pw))
            mx.eval(self.dus_ranks[key])
        return self.dus_ranks[key]

    def update_mask(self, shape: tuple[int, ...], target_pt: float, mask_schedule: str, pt: int, ph: int, pw: int) -> mx.array:
        if mask_schedule == "random":
            return mx.random.uniform(shape=shape) < target_pt
        if mask_schedule == "dus":
            return self.dus_rank(pt, ph, pw) < target_pt
        raise ValueError(f"unsupported mask schedule {mask_schedule}")

    def refinement_update(
        self,
        logits: mx.array,
        pure_rand: mx.array,
        *,
        target_pt: float,
        token_count: int,
        d: int,
        pt: int,
        ph: int,
        pw: int,
        sampling_mode: str = "categorical",
        mask_schedule: str = "random",
    ) -> tuple[mx.array, mx.array]:
        if self.compile_refinement_update and sampling_mode == "categorical" and mask_schedule == "random":
            return self.refinement_update_compiled(logits, pure_rand, target_pt, token_count, d, pt, ph, pw)
        pred_labels = self.sample_labels(logits, token_count, d, pt, ph, pw, sampling_mode)
        mask = self.update_mask(pred_labels.shape, target_pt, mask_schedule, pt, ph, pw)
        mixed = mx.where(mask, pred_labels, pure_rand).astype(mx.int32)
        return pred_labels, mixed

    def sample_labels(
        self,
        logits: mx.array,
        token_count: int,
        d: int,
        pt: int,
        ph: int,
        pw: int,
        sampling_mode: str,
    ) -> mx.array:
        if sampling_mode == "categorical":
            sample = mx.random.categorical(logits.reshape(-1, self.config.detail_num_lvl)).reshape(1, token_count, d)
        elif sampling_mode == "binary":
            if self.config.detail_num_lvl != 2:
                raise ValueError("binary sampling requires detail_num_lvl=2")
            flat = logits.reshape(-1, 2).astype(mx.float32)
            prob_one = mx.sigmoid(flat[:, 1] - flat[:, 0])
            sample = (mx.random.uniform(shape=prob_one.shape) < prob_one).astype(mx.int32).reshape(1, token_count, d)
        elif sampling_mode == "argmax":
            sample = mx.argmax(logits, axis=-1).reshape(1, token_count, d)
        else:
            raise ValueError(f"unsupported sampling mode {sampling_mode}")
        labels = sample.reshape(1, pt, ph, pw, d)
        return mx.transpose(labels, (0, 4, 1, 2, 3)).astype(mx.int32)

    def refinement_update_compiled(
        self,
        logits: mx.array,
        pure_rand: mx.array,
        target_pt: float,
        token_count: int,
        d: int,
        pt: int,
        ph: int,
        pw: int,
    ) -> tuple[mx.array, mx.array]:
        key = (int(token_count), int(d), int(pt), int(ph), int(pw))
        classes = self.config.detail_num_lvl

        def update(logits_: mx.array, pure_rand_: mx.array, target_: mx.array) -> tuple[mx.array, mx.array]:
            sample = mx.random.categorical(logits_.reshape(-1, classes)).reshape(1, token_count, d)
            labels = sample.reshape(1, pt, ph, pw, d)
            labels = mx.transpose(labels, (0, 4, 1, 2, 3)).astype(mx.int32)
            mask = mx.random.uniform(shape=labels.shape) < target_
            mixed = mx.where(mask, labels, pure_rand_).astype(mx.int32)
            return labels, mixed

        if key not in self.compiled_refinement_updates:
            try:
                self.compiled_refinement_updates[key] = mx.compile(update, state=[mx.random.state])
            except TypeError:
                self.compiled_refinement_updates[key] = None
        target = mx.array(target_pt, dtype=mx.float32)
        compiled = self.compiled_refinement_updates[key]
        return update(logits, pure_rand, target) if compiled is None else compiled(logits, pure_rand, target)

    def close(self) -> None:
        self.w.clear()
        self.w32.clear()
        self.w_compute.clear()
        self.w_quant.clear()
        self.w_mlp_gate_up.clear()
        self.text_kv_cache.clear()
        self.compiled_text_blocks.clear()
        self.compiled_visual_blocks.clear()
        self.compiled_cfg_visual_pass = None
        self.compiled_cfg_logits_passes.clear()
        self.compiled_refinement_updates.clear()
        self.dus_ranks.clear()
        self._word_embed_label_base = None
        self._word_embed_label_delta = None
        mx.clear_cache()

    def refine(
        self,
        cond_text: mx.array,
        uncond_text: mx.array | None,
        *,
        pt: int,
        ph: int,
        pw: int,
        mapped_h_div_w: float,
        steps: int,
        guidance: float,
        temperature: float,
        seed: int,
        snr_shift: float = 1.0,
        progress: Callable[[dict], None] | None = None,
        cond_cache_key: str | None = None,
        uncond_cache_key: str | None = None,
        detailed_stats: bool = False,
        exact_step_sync: bool = False,
        sampling_mode: str = "categorical",
        mask_schedule: str = "random",
        min_change_frac: float = 0.0,
        track_token_confidence: bool = False,
        precompute_pt_embed: bool = False,
        cfg_start_step: int = 0,
        frame_callback: Callable[[int, mx.array], None] | None = None,
        capture_interval: int = 0,
    ) -> tuple[mx.array, list[dict]]:
        if sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"unsupported sampling mode {sampling_mode}")
        if mask_schedule not in MASK_SCHEDULES:
            raise ValueError(f"unsupported mask schedule {mask_schedule}")
        if min_change_frac < 0.0:
            raise ValueError("min_change_frac must be non-negative")
        if cfg_start_step < 0:
            raise ValueError("cfg_start_step must be non-negative")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        mx.random.seed(seed)
        cond_cache = self.encode_text_cache_cached(cond_text, cond_cache_key)
        uncond_cache = (
            self.encode_text_cache_cached(uncond_text, uncond_cache_key) if uncond_text is not None and guidance != 1.0 else None
        )
        d = self.config.vae_latent_dim * self.config.hbq_round
        pure_rand = mx.random.randint(0, self.config.detail_num_lvl, (1, d, pt, ph, pw), dtype=mx.int32)
        mixed = pure_rand
        next_pt = 0.0
        stats: list[dict] = []
        pred_labels = pure_rand
        token_count = pt * ph * pw
        rope = mx.concatenate([visual_rope(pt, ph, pw, mapped_h_div_w), text_rope(1, offset=512)], axis=1)
        mx.eval(rope)
        cfg_cache = self.make_cfg_cache(cond_cache, uncond_cache, token_count + 1) if uncond_cache is not None else None
        target_pts = [refinement_target_pt(step, steps, snr_shift) for step in range(steps)]
        pt_embed_tokens = None
        if precompute_pt_embed and not exact_step_sync:
            pt_embed_tokens = [self.pt_embed(0.0)] + [self.pt_embed(target_pts[s]) for s in range(steps - 1)]
            mx.eval(*pt_embed_tokens)
        prev_pred_labels: mx.array | None = None
        stable_count = 0
        for step in range(steps):
            cur_pt = next_pt
            pt_embed_token = self.pt_embed(cur_pt) if pt_embed_tokens is None else pt_embed_tokens[step]
            visual_input = mx.concatenate([self.embed_visual_labels(mixed), pt_embed_token], axis=1)
            # When cfg_start_step > 0, run only the conditional (cond-only) visual
            # forward for steps < cfg_start_step, and resume the CFG-batched path
            # at step >= cfg_start_step. Bit-identical to today when K=0 because
            # the condition reduces to (uncond_cache is not None) for all steps.
            use_cfg_this_step = uncond_cache is not None and step >= cfg_start_step
            if use_cfg_this_step:
                if self.compile_cfg_logits:
                    logits = self.cfg_logits_compiled(visual_input, rope, cfg_cache, token_count, guidance, temperature)
                else:
                    cfg_out = self.visual_forward_cfg_batched(visual_input, rope, cfg_cache)
                    cfg_logits = self.logits(cfg_out, token_count)
                    cond_logits = cfg_logits[:1]
                    uncond_logits = cfg_logits[1:2]
                    logits = uncond_logits + guidance * (cond_logits - uncond_logits)
                    logits = logits / float(temperature)
            else:
                cond_logits = self.logits(self.visual_forward_embedded(visual_input, rope, cond_cache), token_count)
                logits = cond_logits / float(temperature)
            entropy_value = None
            if detailed_stats:
                probs = mx.softmax(logits, axis=-1)
                probs32 = probs.astype(mx.float32)
                entropy = (-probs32 * mx.log2(mx.maximum(probs32, mx.array(1e-12, dtype=mx.float32)))).sum(axis=-1).mean()
                entropy_value = float(entropy.item())
            confidence_stats: dict[str, float] = {}
            if track_token_confidence:
                probs = mx.softmax(logits.reshape(-1, self.config.detail_num_lvl), axis=-1)
                token_conf = mx.max(probs, axis=-1)
                mean_conf = mx.mean(token_conf.astype(mx.float32))
                pct_confident_90 = mx.mean((token_conf > 0.9).astype(mx.float32))
                mx.eval(mean_conf, pct_confident_90)
                confidence_stats = {
                    "mean_confidence": float(mean_conf.item()),
                    "pct_confident_90": float(pct_confident_90.item()),
                }
            target_pt = target_pts[step]
            if exact_step_sync:
                pred_labels = self.sample_labels(logits, token_count, d, pt, ph, pw, sampling_mode)
                mask = self.update_mask(pred_labels.shape, target_pt, mask_schedule, pt, ph, pw)
                mixed = mx.where(mask, pred_labels, pure_rand).astype(mx.int32)
                next_pt = float(mx.mean(mask.astype(mx.float32)).item())
            else:
                pred_labels, mixed = self.refinement_update(
                    logits,
                    pure_rand,
                    target_pt=target_pt,
                    token_count=token_count,
                    d=d,
                    pt=pt,
                    ph=ph,
                    pw=pw,
                    sampling_mode=sampling_mode,
                    mask_schedule=mask_schedule,
                )
                next_pt = target_pt
            stat = {
                "step": step + 1,
                "cur_pt": float(cur_pt),
                "next_pt": next_pt,
                "target_pt": target_pt,
                "entropy": entropy_value,
            }
            stat.update(confidence_stats)
            stats.append(stat)
            mx.eval(mixed)
            early_stop = False
            if min_change_frac > 0.0:
                if prev_pred_labels is not None:
                    changed = mx.mean((pred_labels != prev_pred_labels).astype(mx.float32))
                    mx.eval(changed)
                    change_frac = float(changed.item())
                    stable_count = stable_count + 1 if change_frac < min_change_frac else 0
                    stat["change_frac"] = change_frac
                    stat["stable_count"] = stable_count
                    if stable_count >= 3:
                        stat["early_stop"] = True
                        early_stop = True
                else:
                    stat["change_frac"] = None
                    stat["stable_count"] = stable_count
                prev_pred_labels = pred_labels
            if progress:
                progress(stat)
            if frame_callback and capture_interval > 0 and (step + 1) % capture_interval == 0:
                raw_frame = self.bit_labels_to_raw(pred_labels)
                mx.eval(raw_frame)
                frame_callback(step + 1, raw_frame)
            if early_stop:
                break
        return self.bit_labels_to_raw(pred_labels), stats
