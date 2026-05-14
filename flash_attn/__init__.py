"""Small flash-attn compatibility shim for local Metal/MPS runs.

The upstream GRN code imports ``flash_attn_varlen_func`` unconditionally even
when it is configured for non-CUDA inference.  Apple Silicon cannot install the
real CUDA extension, so this module provides the subset GRN needs and delegates
to PyTorch scaled-dot-product attention.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    **_: object,
) -> torch.Tensor:
    """CPU/MPS fallback with the same shape contract used by GRN.

    Inputs are packed as ``[total_tokens, heads, head_dim]``.  Cumulative
    sequence lengths split conditional/unconditional chunks.  The output keeps
    the same packed layout.
    """

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("expected q/k/v tensors shaped [tokens, heads, dim]")
    return_float32 = q.device.type != "cuda" and q.dtype == torch.bfloat16
    if q.device.type != "cuda" and q.dtype == torch.bfloat16:
        dtype = torch.float16 if q.device.type == "mps" else torch.float32
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)

    q_lens = cu_seqlens_q.detach().to("cpu", torch.long).tolist()
    k_lens = cu_seqlens_k.detach().to("cpu", torch.long).tolist()
    if len(q_lens) != len(k_lens):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have same length")

    outputs = []
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    for i in range(len(q_lens) - 1):
        qs, qe = q_lens[i], q_lens[i + 1]
        ks, ke = k_lens[i], k_lens[i + 1]
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, dropout_p=0.0)
        outputs.append(out.squeeze(0).transpose(0, 1))
    packed = torch.cat(outputs, dim=0)
    return packed.float() if return_float32 else packed
