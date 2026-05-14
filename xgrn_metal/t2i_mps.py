from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_GRN = (ROOT / "../GRN").resolve()


def ensure_reference_importable() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    root = str(ROOT)
    grn = str(REFERENCE_GRN)
    if root not in sys.path:
        sys.path.insert(0, root)
    if grn not in sys.path:
        sys.path.insert(1, grn)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def runtime_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"mps", "cuda"}:
        return torch.float16
    return torch.float32


def parse_dtype(name: str | None, device: torch.device) -> torch.dtype:
    if name is None or name == "auto":
        return runtime_dtype(device)
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name}")


def default_args(task: str, pn: str, device: torch.device, model_dir: Path) -> SimpleNamespace:
    args = SimpleNamespace()
    args.video_frames = 81
    args.model_path = str(model_dir / ("GRN_T2I_2B.pth" if task.upper() == "T2I" else "GRN_T2V_2B.pth"))
    args.vae_path = str(model_dir / "HBQ_tokenizer_64dim_M4.ckpt")
    args.text_encoder_ckpt = str(model_dir / "umt5-xxl")
    args.cfg = 1
    args.fps = 16
    args.cfg_insertion_layer = 0
    args.vae_latent_dim = 64
    args.hbq_round = 4
    args.rope_type = "3d"
    args.num_lvl = 2
    args.model = "GRN2b"
    args.rope2d_normalized_by_hw = 2
    args.text_channels = 4096
    args.apply_spatial_patchify = 0
    args.h_div_w_template = 1.0
    args.cache_dir = "/tmp"
    args.checkpoint_type = "torch"
    args.seed = 42
    args.bf16 = 1
    args.dynamic_scale_schedule = "GRN_vae_stride16"
    args.train_h_div_w_list = "[]"
    args.max_infer_steps = 50
    args.min_infer_steps = 50
    args.video_caption_type = "tarsier2_caption"
    args.temporal_compress_rate = 4
    args.cached_video_frames = 81
    args.duration_resolution = 0.25
    args.video_fps = 16
    args.simple_text_proj = 1
    args.min_duration = -1
    args.fsdp_save_flatten_model = 1
    args.use_learnable_dim_proj = 0
    args.use_fsq_cls_head = 1
    args.use_feat_proj = 0
    args.use_clipwise_caption = 0
    args.use_ada_layer_norm = 0
    args.cfg_type = "cfg_interval_0.0"
    args.add_scale_token = 1
    args.vae_encoder_out_type = "feature_tanh"
    args.alpha = 1004
    args.refine_mode = "ar_discrete_GRN_bit"
    args.add_class_token = 0
    args.resample_rand_labels_per_step = 0
    args.cfg_val = 3.0
    args.scale_repetition = ""
    args.gt_leak = -1
    args.use_refined_prompt = None
    args.use_prompt_engineering = 0
    args.quality_prompt = ""
    args.meta = ""
    args.train_split_file = ""
    args.n_sampes = 1
    args.other_device = device
    args.task = task.upper()
    args.pn = pn
    args.max_duration = (args.video_frames - 1) / 4
    args.num_of_label_value = args.num_lvl
    args.semantic_num_lvl = args.num_lvl
    args.detail_num_lvl = args.num_lvl
    args.semantic_scale_dim = args.vae_latent_dim
    args.detail_scale_dim = args.vae_latent_dim
    return args


def _cache_dir(model_dir: Path) -> Path:
    path = model_dir / ".xgrn_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _quantized_text_encoder_path(args: SimpleNamespace, dtype: torch.dtype) -> Path:
    suffix = "fp16" if dtype == torch.float16 else "fp32"
    return _cache_dir(Path(args.text_encoder_ckpt).parents[0]) / f"umt5-xxl-encoder-{suffix}.pth"


def _ensure_text_encoder_checkpoint(
    args: SimpleNamespace,
    dtype: torch.dtype,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    src = Path(args.text_encoder_ckpt) / "models_t5_umt5-xxl-enc-bf16.pth"
    if dtype != torch.float16:
        return src
    dst = _quantized_text_encoder_path(args, dtype)
    meta = dst.with_suffix(".json")
    src_stat = src.stat()
    expected = {
        "source": str(src),
        "source_size": src_stat.st_size,
        "source_mtime_ns": src_stat.st_mtime_ns,
        "dtype": "float16",
    }
    if dst.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()) == expected:
                return dst
        except json.JSONDecodeError:
            pass
    if progress:
        progress("quantizing UMT5 text encoder checkpoint to fp16 cache")
    state = torch.load(src, map_location="cpu", mmap=True)
    state = {k: v.to(dtype=torch.float16) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in state.items()}
    tmp = dst.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(dst)
    meta.write_text(json.dumps(expected, indent=2))
    del state
    gc.collect()
    return dst


def _text_embedding_cache_path(
    args: SimpleNamespace,
    prompt: str,
    negative_prompt: str,
    dtype: torch.dtype,
) -> Path:
    ckpt = Path(args.text_encoder_ckpt) / "models_t5_umt5-xxl-enc-bf16.pth"
    stat = ckpt.stat()
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "dtype": str(dtype),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "text_len": 512,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return _cache_dir(Path(args.text_encoder_ckpt).parents[0]) / f"text-cond-{digest}.pt"


@dataclass
class GenerationResult:
    image: Image.Image
    video: Optional[Path]
    frames: list[Image.Image]
    stats: list[dict[str, Any]]
    model_dir: Path
    elapsed_sec: float


def download_weights(
    model_dir: Path,
    repo_id: str = "bytedance-research/GRN",
    include_t2v: bool = True,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "GRN_T2I_2B.pth",
        "HBQ_tokenizer_64dim_M4.ckpt",
        "umt5-xxl/**",
        "README.md",
    ]
    if include_t2v:
        patterns.append("GRN_T2V_2B.pth")
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(model_dir),
            allow_patterns=patterns,
        )
    )


def _load_text_conditions(
    prompt: str,
    negative_prompt: str,
    args: SimpleNamespace,
    device: torch.device,
    dtype: torch.dtype,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[tuple[torch.Tensor, list[int], torch.Tensor, int], tuple[torch.Tensor, list[int], torch.Tensor, int]]:
    ensure_reference_importable()
    cache_path = _text_embedding_cache_path(args, prompt, negative_prompt, dtype)
    if cache_path.exists():
        if progress:
            progress("loading cached text embeddings")
        cached = torch.load(cache_path, map_location="cpu")
        cond = (
            cached["cond_kv"].to(device=device, dtype=dtype),
            list(cached["cond_lens"]),
            cached["cond_cu"].to(device=device, dtype=torch.int32),
            int(cached["cond_max"]),
        )
        uncond = (
            cached["uncond_kv"].to(device=device, dtype=dtype),
            list(cached["uncond_lens"]),
            cached["uncond_cu"].to(device=device, dtype=torch.int32),
            int(cached["uncond_max"]),
        )
        return cond, uncond

    original_current_device = torch.cuda.current_device
    if not torch.cuda.is_available():
        torch.cuda.current_device = lambda: "cpu"  # type: ignore[assignment]
    try:
        from grn.models.umt5.t5 import T5EncoderModel
    finally:
        torch.cuda.current_device = original_current_device

    if progress:
        progress("loading UMT5 text encoder")
    checkpoint_path = _ensure_text_encoder_checkpoint(args, dtype, progress)
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(Path(args.text_encoder_ckpt) / "umt5-xxl"),
        enable_fsdp=False,
    )

    def encode_one(text: str) -> tuple[torch.Tensor, list[int], torch.Tensor, int]:
        if progress:
            progress(f"encoding prompt: {text[:80]}")
        with torch.no_grad():
            features = text_encoder([text], device)
        lens = [len(item) for item in features]
        cu_seqlens = [0]
        for length in lens:
            cu_seqlens.append(cu_seqlens[-1] + length)
        kv_compact = torch.cat(features, dim=0).to(device=device, dtype=dtype)
        return kv_compact, lens, torch.tensor(cu_seqlens, dtype=torch.int32, device=device), max(lens)

    cond = encode_one(prompt)
    uncond = encode_one(negative_prompt)
    torch.save(
        {
            "cond_kv": cond[0].detach().cpu().to(dtype=dtype),
            "cond_lens": cond[1],
            "cond_cu": cond[2].detach().cpu(),
            "cond_max": cond[3],
            "uncond_kv": uncond[0].detach().cpu().to(dtype=dtype),
            "uncond_lens": uncond[1],
            "uncond_cu": uncond[2].detach().cpu(),
            "uncond_max": uncond[3],
        },
        cache_path,
    )
    text_encoder.model.to("cpu")
    del text_encoder
    clear_device_cache(device)
    return cond, uncond


def _load_visual_tokenizer(args: SimpleNamespace, device: torch.device, dtype: torch.dtype) -> Any:
    ensure_reference_importable()
    from grn.models.hbq_tokenizer import HBQ_Tokenizer

    vae = HBQ_Tokenizer(args=args, latent_channels=args.detail_scale_dim, encoder_out_type="feature_tanh")
    vae.eval().requires_grad_(False)
    state = torch.load(args.vae_path, map_location="cpu")
    state = state["ema"] if "ema" in state else state["vae"]
    vae.load_state_dict(state, assign=True)
    del state
    return vae.to(device=device, dtype=dtype)


def _load_transformer(args: SimpleNamespace, vae: Any, device: torch.device, dtype: torch.dtype) -> Any:
    ensure_reference_importable()
    from timm.models import create_model

    state = torch.load(args.model_path, map_location="cpu")
    with torch.no_grad():
        model = create_model(
            args.model,
            vae_local=vae,
            text_channels=args.text_channels,
            text_maxlen=512,
            shared_aln=True,
            raw_scale_schedule=None,
            checkpointing="full-block",
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=False,
            num_of_label_value=args.num_of_label_value,
            rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
            pn=args.pn,
            apply_spatial_patchify=args.apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=args.train_h_div_w_list,
            dynamic_scale_schedule=args.dynamic_scale_schedule,
            video_frames=args.video_frames,
            other_args=args,
        )
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=dtype)


def _tensor_images_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    arr = arr[..., ::-1]
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _tensor_video_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim != 5:
        raise ValueError(f"expected [B,T,H,W,C] video tensor, got {arr.shape}")
    return arr[0, ..., ::-1].astype(np.uint8)


def _save_video(img_tensor: torch.Tensor, output_path: Path, fps: int) -> Path:
    import imageio.v3 as iio

    frames = _tensor_video_to_rgb(img_tensor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, frames, fps=fps)
    return output_path


def patch_grn_for_mps() -> None:
    ensure_reference_importable()
    import tqdm
    from grn.models.grn import GRN, bld_to_bthwd
    from grn.schedules.global_refine import shift_pt
    from grn.utils_t2iv.hbq_util_t2iv import bit_label2raw_feature, multiclass_labels2onehot_input

    if getattr(GRN, "_xgrn_mps_patched", False):
        return

    @torch.no_grad()
    def autoregressive_infer_mps(
        self: Any,
        vae: Optional[Any] = None,
        scale_schedule: Optional[list[tuple[int, int, int]]] = None,
        label_B_or_BLT: Optional[list[tuple[torch.Tensor, ...]]] = None,
        negative_label_B_or_BLT: Optional[tuple[torch.Tensor, ...]] = None,
        g_seed: Optional[int] = None,
        cfg_list: Optional[list[float]] = None,
        tau_list: Optional[list[float]] = None,
        gt_leak: int = 0,
        args: Optional[Any] = None,
        get_visual_rope_embeds: Optional[Any] = None,
        context_info: Optional[Any] = None,
        noise_list: Optional[list[torch.Tensor]] = None,
        class_token_id: int = 0,
        uncond_class_token_id: int = 1000,
        capture_interval: int = 0,
        progress_cb: Optional[Callable[[int, int, dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], torch.Tensor]:
        assert scale_schedule is not None
        assert label_B_or_BLT is not None
        assert args is not None
        assert get_visual_rope_embeds is not None
        assert context_info is not None
        assert len(scale_schedule) == 1
        if cfg_list is None:
            cfg_list = []
        elif not isinstance(cfg_list, list):
            cfg_list = [float(cfg_list)] * len(scale_schedule)
        if tau_list is None:
            tau_list = []
        elif not isinstance(tau_list, list):
            tau_list = [float(tau_list)] * len(scale_schedule)
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        rng = torch.Generator(device=args.other_device)
        if g_seed is not None:
            rng.manual_seed(int(g_seed))

        ret, idx_Bl_list = [], []
        for block in self.unregistered_blocks:
            block.attn.kv_caching(True)

        text_scales = len(label_B_or_BLT)
        total_steps = args.max_infer_steps + text_scales
        pbar = tqdm.tqdm(total=total_steps, desc="GRN refine")
        block_chunks = self.block_chunks
        use_cfg = any(np.array(cfg_list) != 1)
        cfg_interval = float(args.cfg_type.split("_")[-1])
        attn_mask = None

        device = torch.device(args.other_device)
        self.rope2d_freqs_grid["freqs_text"] = self.rope2d_freqs_grid["freqs_text"].to(device)

        for si, text_cond_tuple in enumerate(label_B_or_BLT):
            prefix_tokens, lens = self.prepare_text_conditions(text_cond_tuple, negative_label_B_or_BLT, use_cfg)
            device = prefix_tokens.device
            if use_cfg:
                rope_cache = torch.cat(
                    [
                        self.rope2d_freqs_grid["freqs_text"][:, :, :, :, : lens[0]],
                        self.rope2d_freqs_grid["freqs_text"][:, :, :, :, : lens[1]],
                    ],
                    dim=4,
                )
            else:
                rope_cache = self.rope2d_freqs_grid["freqs_text"][:, :, :, :, : lens[0]]
            last_stage = prefix_tokens
            for block in block_chunks:
                last_stage = block(
                    x=last_stage,
                    e0=None,
                    attn_bias_or_two_vector=attn_mask,
                    attn_fn=None,
                    rope2d_freqs_grid=rope_cache,
                    scale_ind=f"t{si}",
                    context_info=context_info,
                    last_diffusion_step=True,
                    ref_text_scale_inds=[],
                    use_cfg=use_cfg,
                    split_cond_uncond=lens,
                )
            pbar.update(1)

        if args.refine_mode == "ar_discrete_GRN_bit":
            classes = 2
            this_scale_var_input = torch.zeros(
                (1, args.detail_scale_dim * args.hbq_round, *scale_schedule[-1]),
                device=prefix_tokens.device,
                dtype=prefix_tokens.dtype,
            )
        else:
            raise ValueError(f"{args.refine_mode=} is not supported by the MPS runner yet")

        scale_token_rope_cache = self.rope2d_freqs_grid["freqs_text"][:, :, :, :, 512 : 512 + args.add_scale_token]
        if noise_list is not None:
            absolute_gt_labels = noise_list[0].to(device).permute(0, 2, 3, 4, 1)

        si = 0
        pt, ph, pw = scale_schedule[0]
        mul_pt_ph_pw = pt * ph * pw
        ref_text_scale_inds = ["t0"]
        cur_round_scales = args.max_infer_steps
        pure_rand_labels = torch.randint(
            low=0,
            high=classes,
            size=this_scale_var_input.shape,
            device=this_scale_var_input.device,
            dtype=this_scale_var_input.dtype,
            generator=rng,
        )
        mixed_xt = pure_rand_labels
        this_scale_var_input = multiclass_labels2onehot_input(mixed_xt, classes).to(dtype=prefix_tokens.dtype)
        next_pt = 0.0
        self.entrophy_statistics.append([])
        self.xgrn_visual_frames = []

        for cur_inner_round_si in range(cur_round_scales):
            cur_pt = next_pt
            scale_token_id = cur_pt
            cfg = cfg_list[0] if cur_pt >= cfg_interval else 1.0
            rope_cache = get_visual_rope_embeds(
                self.rope2d_freqs_grid,
                scale_schedule,
                si,
                0,
                device,
                args,
                context_info,
                args.mapped_h_div_w_template,
            )
            last_stage = self.embeds_codes2input(this_scale_var_input)
            if args.add_scale_token > 0:
                pt_tokens = self.pt_embedder(torch.tensor([scale_token_id], device=device))
                last_stage = torch.cat((last_stage, pt_tokens), dim=1)
                rope_cache = torch.cat((rope_cache, scale_token_rope_cache), dim=4)

            if use_cfg:
                last_stage = torch.cat([last_stage, last_stage], dim=1)
                rope_cache = torch.cat([rope_cache, rope_cache], dim=4)
                split_cond_uncond = [mul_pt_ph_pw + args.add_scale_token] * 2
            else:
                split_cond_uncond = [mul_pt_ph_pw + args.add_scale_token]

            for block in block_chunks:
                last_stage = block(
                    x=last_stage,
                    e0=None,
                    attn_bias_or_two_vector=attn_mask,
                    attn_fn=None,
                    rope2d_freqs_grid=rope_cache,
                    scale_ind=si,
                    context_info=context_info,
                    last_diffusion_step=False,
                    ref_text_scale_inds=ref_text_scale_inds,
                    use_cfg=use_cfg,
                    split_cond_uncond=split_cond_uncond,
                )

            logits = self.get_logits_during_infer(last_stage, e=None)
            tmp_bs, tmp_seq_len = logits.shape[:2]
            logits = logits.reshape(tmp_bs, tmp_seq_len, -1, args.detail_num_lvl)
            pred_cond_logits = logits[:, :mul_pt_ph_pw]
            pred_cond_probs = pred_cond_logits.softmax(-1)
            categories = pred_cond_logits.shape[-1]
            entropy = (-pred_cond_probs * torch.log2(pred_cond_probs.clamp_min(1e-12))).sum(-1).mean().item()
            entropy = entropy / np.log2(categories)

            pt_unshift = (cur_inner_round_si + 1) / max(1, args.complexity_aware_Tmax - 1)
            pt_shift = shift_pt(min(1.0, pt_unshift), args.snr_shift)
            next_pt = float(1 - np.cos(np.pi / 2 * pt_shift))
            next_pt = next_pt * 0.999

            pred_cond_labels = torch.argmax(pred_cond_probs, dim=-1)
            pred_cond_labels = bld_to_bthwd(pred_cond_labels, pt, ph, pw)
            pred_uncond_logits = logits[:, (mul_pt_ph_pw + args.add_scale_token) : (2 * mul_pt_ph_pw + args.add_scale_token)]
            pred_cfg_logits = pred_uncond_logits + cfg * (pred_cond_logits - pred_uncond_logits) if cfg != 1 else pred_cond_logits
            pred_cfg_probs = pred_cfg_logits.mul(1 / tau_list[si]).softmax(dim=-1)
            pred_cfg_labels = torch.argmax(pred_cfg_probs, dim=-1)
            pred_cfg_labels = bld_to_bthwd(pred_cfg_labels, pt, ph, pw)
            pred_sample_labels = torch.multinomial(
                pred_cfg_probs.reshape(-1, args.detail_num_lvl),
                num_samples=1,
                replacement=True,
                generator=rng,
            ).view(tmp_bs, mul_pt_ph_pw, -1)
            pred_sample_probs = torch.gather(pred_cfg_probs, dim=3, index=pred_sample_labels.unsqueeze(-1)).squeeze(-1)
            pred_sample_probs = bld_to_bthwd(pred_sample_probs, pt, ph, pw)
            pred_sample_labels = bld_to_bthwd(pred_sample_labels, pt, ph, pw)

            assume_flip_ratio = (1 - cur_pt) / args.detail_num_lvl * 100.0
            mixed_xt_Bthwd_01 = mixed_xt.clone().permute(0, 2, 3, 4, 1)
            mixed_xt_Bthwd_01[mixed_xt_Bthwd_01 < 0] = 0
            pred_cond_flip_ratio = (pred_cond_labels != mixed_xt_Bthwd_01).sum() / pred_cond_labels.numel() * 100.0
            pred_cfg_flip_ratio = (pred_cfg_labels != mixed_xt_Bthwd_01).sum() / pred_cfg_labels.numel() * 100.0
            pred_sample_flip_ratio = (pred_sample_labels != mixed_xt_Bthwd_01).sum() / pred_sample_labels.numel() * 100.0
            pred_zero_ratio = (pred_cond_labels == 0).sum() / pred_cond_labels.numel() * 100.0
            pred_one_ratio = (pred_cond_labels == 1).sum() / pred_cond_labels.numel() * 100.0

            stat = {
                "step": cur_inner_round_si + 1,
                "cur_pt": float(cur_pt),
                "next_pt": float(next_pt),
                "entropy": float(entropy),
                "assume_flip_ratio": float(assume_flip_ratio),
                "pred_cond_flip_ratio": float(pred_cond_flip_ratio.item()),
                "pred_cfg_flip_ratio": float(pred_cfg_flip_ratio.item()),
                "pred_sample_flip_ratio": float(pred_sample_flip_ratio.item()),
                "pred_zero_ratio": float(pred_zero_ratio.item()),
                "pred_one_ratio": float(pred_one_ratio.item()),
                "cfg": float(cfg),
            }
            self.entrophy_statistics[-1].append(stat)
            if progress_cb:
                progress_cb(cur_inner_round_si + 1, cur_round_scales, stat)

            if cur_inner_round_si < gt_leak:
                gt_labels = absolute_gt_labels
                pred_sample_labels = gt_labels

            pred_sample_labels = pred_sample_labels.permute(0, 4, 1, 2, 3)
            pred_sample_probs = pred_sample_probs.permute(0, 4, 1, 2, 3)
            use_predict_mask = torch.rand(pred_sample_labels.shape, device=device, generator=rng) < next_pt
            next_pt = use_predict_mask.float().mean().item()
            mixed_xt = torch.where(use_predict_mask, pred_sample_labels, pure_rand_labels)
            this_scale_var_input = multiclass_labels2onehot_input(mixed_xt, classes).to(dtype=prefix_tokens.dtype)
            pbar.update(1)

            if capture_interval and (
                (cur_inner_round_si + 1) % capture_interval == 0 or cur_inner_round_si + 1 == cur_round_scales
            ):
                approx_signal_i = bit_label2raw_feature(pred_sample_labels, hbq_round=args.hbq_round)
                approx_signal_i = approx_signal_i.to(device=device, dtype=prefix_tokens.dtype)
                img_i = self.summed_codes2images(vae, approx_signal_i)
                self.xgrn_visual_frames.append(_tensor_images_to_pil(img_i))

            if abs(cur_pt - 1) < 1e-6:
                break

        approx_signal = bit_label2raw_feature(pred_sample_labels, hbq_round=args.hbq_round)
        approx_signal = approx_signal.to(device=device, dtype=prefix_tokens.dtype)
        for block in self.unregistered_blocks:
            block.attn.kv_caching(False)
        img = self.summed_codes2images(vae, approx_signal)
        return ret, idx_Bl_list, img

    GRN.autoregressive_infer = autoregressive_infer_mps
    GRN._xgrn_mps_patched = True


def generate_t2i(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    pn: str = "0.06M",
    steps: int = 16,
    guidance_scale: float = 3.0,
    temperature: float = 1.1,
    h_div_w: float = 1.0,
    snr_shift: float = 1.0,
    capture_interval: int = 0,
    model_dir: Path | str = ROOT / "models/GRN",
    device_name: str = "auto",
    dtype_name: str = "auto",
    progress: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    return generate(
        task="T2I",
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        pn=pn,
        steps=steps,
        guidance_scale=guidance_scale,
        temperature=temperature,
        h_div_w=h_div_w,
        duration=0.0,
        snr_shift=snr_shift,
        capture_interval=capture_interval,
        model_dir=model_dir,
        device_name=device_name,
        dtype_name=dtype_name,
        progress=progress,
    )


def generate(
    task: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    pn: str = "0.06M",
    steps: int = 16,
    guidance_scale: float = 3.0,
    temperature: float = 1.1,
    h_div_w: float = 1.0,
    duration: float = 0.0,
    snr_shift: float = 1.0,
    capture_interval: int = 0,
    model_dir: Path | str = ROOT / "models/GRN",
    device_name: str = "auto",
    dtype_name: str = "auto",
    progress: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    ensure_reference_importable()
    patch_grn_for_mps()
    from grn.schedules import get_encode_decode_func
    from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index

    start = time.time()
    task = task.upper()
    if task not in {"T2I", "T2V"}:
        raise ValueError(f"unsupported task: {task}")
    model_dir = Path(model_dir)
    device = pick_device(device_name)
    dtype = parse_dtype(dtype_name, device)
    if progress:
        progress(f"using device={device}, dtype={dtype}")

    download_weights(model_dir, include_t2v=task == "T2V")
    args = default_args(task, pn, device, model_dir)
    args.seed = int(seed)
    args.cfg_val = float(guidance_scale)
    args.tau = float(temperature)
    args.complexity_aware_Tmin = min(int(steps), 50)
    args.complexity_aware_Tmax = max(2, int(steps))
    args.complexity_aware_k = 0
    args.complexity_aware_b = int(steps)
    args.complexity_aware_wp = 5
    args.snr_shift = float(snr_shift)
    args.max_infer_steps = max(2, int(steps))
    args.min_infer_steps = max(2, int(steps))
    args.other_device = device

    _, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.train_h_div_w_list, args.video_frames
    )
    mapped_h_div_w = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    args.mapped_h_div_w_template = mapped_h_div_w
    if task == "T2I":
        num_frames = 1
        duration = 0.0
    else:
        num_frames = min(args.video_frames, max(5, int(duration * args.video_fps + 1)))
    pt_key = (num_frames - 1) // 4 + 1
    scale_schedule = dynamic_resolution_h_w[mapped_h_div_w][args.pn]["pt2scale_schedule"][pt_key]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))

    cond, uncond = _load_text_conditions(prompt, negative_prompt, args, device, dtype, progress)
    if progress:
        progress("loading visual tokenizer")
    vae = _load_visual_tokenizer(args, device, dtype)
    clear_device_cache(device)
    if progress:
        progress("loading GRN transformer")
    model = _load_transformer(args, vae, device, dtype)
    clear_device_cache(device)

    status = {"last": ""}

    def progress_cb(step: int, total: int, stat: dict[str, Any]) -> None:
        status["last"] = f"refine {step}/{total}: entropy={stat['entropy']:.3f}, signal={stat['next_pt']:.3f}"
        if progress:
            progress(status["last"])

    if progress:
        progress("running refinement")
    _, _, img_tensor = model.autoregressive_infer(
        vae=vae,
        scale_schedule=scale_schedule,
        label_B_or_BLT=[cond],
        negative_label_B_or_BLT=uncond,
        g_seed=int(seed),
        cfg_list=float(guidance_scale),
        tau_list=float(temperature),
        gt_leak=args.gt_leak,
        args=args,
        get_visual_rope_embeds=get_visual_rope_embeds,
        context_info=context_info,
        noise_list=None,
        class_token_id=0,
        capture_interval=int(capture_interval),
        progress_cb=progress_cb,
    )
    image = _tensor_images_to_pil(img_tensor)
    frames = getattr(model, "xgrn_visual_frames", [])
    stats = model.entrophy_statistics[-1] if model.entrophy_statistics else []
    elapsed = time.time() - start

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    image_path = out_dir / ("latest_t2i.png" if task == "T2I" else "latest_t2v_first_frame.png")
    image.save(image_path)
    video_path = None
    if task == "T2V":
        video_path = _save_video(img_tensor, out_dir / "latest_t2v.mp4", args.fps)
    if frames:
        frames[0].save(
            out_dir / "latest_refinement.gif",
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0,
        )

    model.to("cpu")
    vae.to("cpu")
    del model, vae
    clear_device_cache(device)
    return GenerationResult(image=image, video=video_path, frames=frames, stats=stats, model_dir=model_dir, elapsed_sec=elapsed)


def save_stats_csv(stats: Iterable[dict[str, Any]]) -> Path:
    import csv

    out = ROOT / "outputs/refinement_stats.csv"
    out.parent.mkdir(exist_ok=True)
    rows = list(stats)
    with out.open("w", newline="") as f:
        if not rows:
            return out
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out
