from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _pad5(x: mx.array, pad_t_left: int, pad_h: int, pad_w: int) -> mx.array:
    if pad_t_left == 0 and pad_h == 0 and pad_w == 0:
        return x
    return mx.pad(x, [(0, 0), (0, 0), (pad_t_left, 0), (pad_h, pad_h), (pad_w, pad_w)])


def _pad4(x: mx.array, pad_h: int, pad_w: int) -> mx.array:
    if pad_h == 0 and pad_w == 0:
        return x
    return mx.pad(x, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])


class HBQMLXDecoder:
    """Native MLX decoder for the fixed official GRN HBQ tokenizer."""

    def __init__(
        self,
        model_dir: Path = Path("models/GRN"),
        weights_dtype: str = "fp16",
        compute_dtype: str = "fp16",
    ):
        if weights_dtype not in {"fp32", "fp16"}:
            raise ValueError(f"unsupported HBQ weights dtype {weights_dtype}")
        if compute_dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"unsupported HBQ compute dtype {compute_dtype}")
        root = "mlx_fp32" if weights_dtype == "fp32" else "mlx"
        suffix = "fp32" if weights_dtype == "fp32" else "fp16"
        path = model_dir / root / f"hbq_{suffix}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"missing HBQ MLX artifact: {path}")
        self.w = mx.load(str(path))
        self.compute_dtype = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}[compute_dtype]
        self.cache: dict[str, mx.array] = {}

    def weight(self, name: str) -> mx.array:
        if self.compute_dtype == self.w[name].dtype:
            return self.w[name]
        key = f"{name}:{self.compute_dtype}"
        if key not in self.cache:
            self.cache[key] = self.w[name].astype(self.compute_dtype)
            mx.eval(self.cache[key])
        return self.cache[key]

    def weight_fp32(self, name: str) -> mx.array:
        key = f"{name}:fp32"
        if key not in self.cache:
            self.cache[key] = self.w[name].astype(mx.float32)
            mx.eval(self.cache[key])
        return self.cache[key]

    def conv3d(self, x: mx.array, prefix: str, cache_x: mx.array | None = None) -> mx.array:
        w = self.weight(f"{prefix}.weight")
        bias = self.weight(f"{prefix}.bias")
        out_c, _in_c, kt, kh, kw = w.shape
        pad_t = 1 if kt == 3 else 0
        pad_h = 1 if kh == 3 else 0
        pad_w = 1 if kw == 3 else 0
        if cache_x is not None and pad_t:
            x = mx.concatenate([cache_x.astype(x.dtype), x], axis=2)
            pad_t_left = max(0, 2 * pad_t - cache_x.shape[2])
        else:
            pad_t_left = 2 * pad_t
        x = _pad5(x, pad_t_left, pad_h, pad_w)
        # MLX convolutions are channel-last.
        x_cl = mx.transpose(x.astype(self.compute_dtype), (0, 2, 3, 4, 1))
        w_cl = mx.transpose(w, (0, 2, 3, 4, 1))
        y = mx.conv3d(x_cl, w_cl, stride=1, padding=0)
        y = y + bias.reshape(1, 1, 1, 1, out_c)
        return mx.transpose(y, (0, 4, 1, 2, 3)).astype(mx.float32)

    def conv2d(self, x: mx.array, prefix: str, padding: int = 0) -> mx.array:
        w = self.weight(f"{prefix}.weight")
        bias = self.weight(f"{prefix}.bias")
        out_c, _in_c, kh, kw = w.shape
        x = _pad4(x, padding, padding)
        x_cl = mx.transpose(x.astype(self.compute_dtype), (0, 2, 3, 1))
        w_cl = mx.transpose(w, (0, 2, 3, 1))
        y = mx.conv2d(x_cl, w_cl, stride=1, padding=0)
        y = y + bias.reshape(1, 1, 1, out_c)
        return mx.transpose(y, (0, 3, 1, 2)).astype(mx.float32)

    def rms_norm(self, x: mx.array, prefix: str) -> mx.array:
        gamma = self.weight_fp32(f"{prefix}.gamma")
        xf = x.astype(mx.float32)
        denom = mx.sqrt(mx.sum(xf * xf, axis=1, keepdims=True) + 1e-12)
        return (xf / denom) * (x.shape[1] ** 0.5) * gamma.astype(mx.float32)

    def residual_block(self, x: mx.array, prefix: str, feat_cache: list[mx.array | str | None] | None = None, feat_idx: list[int] | None = None) -> mx.array:
        shortcut = x
        if f"{prefix}.shortcut.weight" in self.w:
            shortcut = self.conv3d(x, f"{prefix}.shortcut")
        h = self.rms_norm(x, f"{prefix}.residual.0")
        h = silu(h)
        h = self.cached_conv3d(h, f"{prefix}.residual.2", feat_cache, feat_idx)
        h = self.rms_norm(h, f"{prefix}.residual.3")
        h = silu(h)
        h = self.cached_conv3d(h, f"{prefix}.residual.6", feat_cache, feat_idx)
        return h + shortcut

    def cached_conv3d(self, x: mx.array, prefix: str, feat_cache: list[mx.array | str | None] | None, feat_idx: list[int] | None) -> mx.array:
        if feat_cache is None or feat_idx is None:
            return self.conv3d(x, prefix)
        idx = feat_idx[0]
        cache_x = x[:, :, -2:, :, :]
        if cache_x.shape[2] < 2 and isinstance(feat_cache[idx], mx.array):
            cache_x = mx.concatenate([feat_cache[idx][:, :, -1:, :, :].astype(x.dtype), cache_x], axis=2)
        cached = feat_cache[idx] if isinstance(feat_cache[idx], mx.array) else None
        y = self.conv3d(x, prefix, cached)
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        return y

    def attention_block(self, x: mx.array, prefix: str) -> mx.array:
        identity = x
        b, c, t, h, w = x.shape
        y = mx.transpose(x, (0, 2, 1, 3, 4)).reshape(b * t, c, h, w)
        y = self.rms_norm(y, f"{prefix}.norm")
        y = self.conv2d(y, f"{prefix}.to_qkv")
        y = mx.transpose(y, (0, 2, 3, 1)).reshape(b * t, 1, h * w, c * 3)
        q, k, v = mx.split(y, 3, axis=-1)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        y = y.reshape(b * t, h * w, c)
        y = mx.transpose(y, (0, 2, 1)).reshape(b * t, c, h, w)
        y = self.conv2d(y, f"{prefix}.proj")
        y = y.reshape(b, t, c, h, w)
        y = mx.transpose(y, (0, 2, 1, 3, 4))
        return y + identity

    def dup_up3d(self, x: mx.array, in_channels: int, out_channels: int, factor_t: int, factor_s: int, first_chunk: bool) -> mx.array:
        repeats = out_channels * factor_t * factor_s * factor_s // in_channels
        b, _c, t, h, w = x.shape
        x = mx.repeat(x, repeats, axis=1)
        x = x.reshape(b, out_channels, factor_t, factor_s, factor_s, t, h, w)
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = x.reshape(b, out_channels, t * factor_t, h * factor_s, w * factor_s)
        if first_chunk:
            x = x[:, :, factor_t - 1 :, :, :]
        return x

    def upsample2d_conv(self, x: mx.array, prefix: str) -> mx.array:
        b, c, t, h, w = x.shape
        y = mx.transpose(x, (0, 2, 3, 4, 1)).reshape(b * t, h, w, c)
        y = mx.repeat(y, 2, axis=1)
        y = mx.repeat(y, 2, axis=2)
        y = y.reshape(b, t, h * 2, w * 2, c)
        y = mx.transpose(y, (0, 4, 1, 2, 3))
        y2 = mx.transpose(y, (0, 2, 1, 3, 4)).reshape(b * t, c, h * 2, w * 2)
        y2 = self.conv2d(y2, prefix, padding=1)
        y = y2.reshape(b, t, c, h * 2, w * 2)
        return mx.transpose(y, (0, 2, 1, 3, 4))

    def resample_up(self, x: mx.array, prefix: str, temporal: bool, feat_cache: list[mx.array | str | None] | None, feat_idx: list[int] | None) -> mx.array:
        b, c, t, h, w = x.shape
        if temporal and feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -2:, :, :]
                if cache_x.shape[2] < 2 and isinstance(feat_cache[idx], mx.array):
                    cache_x = mx.concatenate([feat_cache[idx][:, :, -1:, :, :].astype(x.dtype), cache_x], axis=2)
                cached = feat_cache[idx] if isinstance(feat_cache[idx], mx.array) else None
                x = self.conv3d(x, f"{prefix}.time_conv", cached)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, t, h, w)
                x = mx.stack([x[:, 0], x[:, 1]], axis=3)
                x = x.reshape(b, c, t * 2, h, w)
        return self.upsample2d_conv(x, f"{prefix}.resample.1")

    def up_block(
        self,
        x: mx.array,
        prefix: str,
        in_dim: int,
        out_dim: int,
        temporal: bool,
        up_flag: bool,
        feat_cache: list[mx.array | str | None] | None,
        feat_idx: list[int] | None,
        first_chunk: bool,
    ) -> mx.array:
        main = x
        for i in range(3):
            block_prefix = f"{prefix}.upsamples.{i}"
            block_in = in_dim if i == 0 else out_dim
            main = self.residual_block(main, block_prefix, feat_cache, feat_idx)
            if i == 0 and block_in != out_dim:
                # The shortcut inside the residual block performs the channel projection.
                pass
        if up_flag:
            main = self.resample_up(main, f"{prefix}.upsamples.3", temporal, feat_cache, feat_idx)
            shortcut = self.dup_up3d(x, in_dim, out_dim, 2 if temporal else 1, 2, first_chunk)
            return main + shortcut
        return main

    def decoder_chunk(
        self,
        x: mx.array,
        feat_cache: list[mx.array | str | None],
        first_chunk: bool,
    ) -> mx.array:
        feat_idx = [0]
        # decoder.conv1
        cache_x = x[:, :, -2:, :, :]
        idx = feat_idx[0]
        cached = feat_cache[idx]
        if cache_x.shape[2] < 2 and isinstance(cached, mx.array):
            cache_x = mx.concatenate([cached[:, :, -1:, :, :].astype(x.dtype), cache_x], axis=2)
        x = self.conv3d(x, "decoder.conv1", cached if isinstance(cached, mx.array) else None)
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        x = self.residual_block(x, "decoder.middle.0", feat_cache, feat_idx)
        x = self.attention_block(x, "decoder.middle.1")
        x = self.residual_block(x, "decoder.middle.2", feat_cache, feat_idx)

        dims = [1024, 1024, 1024, 512, 256]
        temporal = [True, True, False, False]
        for i in range(4):
            x = self.up_block(
                x,
                f"decoder.upsamples.{i}",
                dims[i],
                dims[i + 1],
                temporal[i],
                i != 3,
                feat_cache,
                feat_idx,
                first_chunk,
            )

        x = self.rms_norm(x, "decoder.head.0")
        x = silu(x)
        x = self.cached_conv3d(x, "decoder.head.2", feat_cache, feat_idx)
        return x

    def unpatchify2(self, x: mx.array) -> mx.array:
        b, crq, f, h, w = x.shape
        c = crq // 4
        x = x.reshape(b, c, 2, 2, f, h, w)
        x = mx.transpose(x, (0, 1, 4, 5, 3, 6, 2))
        return x.reshape(b, c, f, h * 2, w * 2)

    def decode_raw(self, z: mx.array) -> mx.array:
        z = z.astype(mx.float32)
        x = self.conv3d(z, "conv2")
        # Match count_conv3d(official_decoder). Shortcut convs are included in
        # the official cache list length even though the forward path does not
        # consume cache entries for shortcuts.
        feat_cache: list[mx.array | str | None] = [None] * 34
        chunks = []
        for i in range(z.shape[2]):
            chunk = self.decoder_chunk(x[:, :, i : i + 1, :, :], feat_cache, first_chunk=(i == 0))
            chunks.append(chunk)
        out = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)
        return self.unpatchify2(out)

    def decode_tensor(self, raw: mx.array) -> mx.array:
        image = self.decode_raw(raw)
        image = mx.clip((image + 1) / 2, 0, 1)
        mx.eval(image)
        return image

    def close(self) -> None:
        self.w.clear()
        self.cache.clear()
        mx.clear_cache()
