from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GRN2BConfig:
    embed_dim: int = 2304
    depth: int = 28
    block_chunks: int = 7
    blocks_per_chunk: int = 4
    num_heads: int = 18
    head_dim: int = 128
    num_key_value_heads: int = 18
    mlp_hidden_dim: int = 8192
    hbq_round: int = 4
    vae_latent_dim: int = 64
    detail_num_lvl: int = 2
    visual_in_dim: int = 512
    text_channels: int = 4096
    add_scale_token: int = 1
    rope_text_len: int = 600
    rope_max_frames: int = 21
    rope_max_height: int = 225
    rope_max_width: int = 225
    rope_base: float = 10000.0
    rope2d_normalized_by_hw: int = 2


GRN2B = GRN2BConfig()


H_DIV_W_TEMPLATES = [
    3.0,
    2.5,
    2.0,
    1.777,
    1.5,
    1.333,
    1.16,
    1.0,
    0.862,
    0.75,
    0.666,
    0.562,
    0.5,
    0.4,
    0.333,
]


PN_BASE_SCALE = {
    "0.06M": 16,
    "0.25M": 32,
    "0.41M": 40,
    "0.92M": 60,
    "1M": 64,
    "2M": 90,
}

