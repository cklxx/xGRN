# 2026-05-15 Fresh First-Run Cache

Goal: verify a real online first-run path with an empty model cache, a separate
HuggingFace home, and a non-7860 app port.

## Environment

- Repo: `/Users/bytedance/code/xGRN`
- Fresh cache: `/tmp/xgrn-firststart-20260515-001232/GRN`
- Fresh HuggingFace home: `/tmp/xgrn-firststart-20260515-001232/hf-home`
- App validation port: `7861`
- Existing 7860 process preserved:
  `85165 /Users/bytedance/code/xGRN/.venv/bin/xgrn-app`

## Commands

```bash
HF_HOME=/tmp/xgrn-firststart-20260515-001232/hf-home \
uv run xgrn-download \
  --model-dir /tmp/xgrn-firststart-20260515-001232/GRN \
  --no-convert

uv run xgrn-convert \
  --model-dir /tmp/xgrn-firststart-20260515-001232/GRN \
  --out-dir /tmp/xgrn-firststart-20260515-001232/GRN/mlx \
  --dtype fp16

uv run xgrn-convert \
  --model-dir /tmp/xgrn-firststart-20260515-001232/GRN \
  --out-dir /tmp/xgrn-firststart-20260515-001232/GRN/mlx_fp32 \
  --dtype fp32

uv run xgrn-app \
  --model-dir /tmp/xgrn-firststart-20260515-001232/GRN \
  --server-port 7861 \
  --bootstrap-only

uv run xgrn-app \
  --model-dir /tmp/xgrn-firststart-20260515-001232/GRN \
  --server-port 7861
curl -fsS http://127.0.0.1:7861/
```

## Timings

| Stage | Start | End | Elapsed |
|---|---|---|---:|
| Download raw HF snapshot | 2026-05-15 00:12:32 CST | 2026-05-15 00:17:28 CST | 296 s |
| Convert MLX fp16 | 2026-05-15 00:17:47 CST | 2026-05-15 00:19:00 CST | 73 s |
| Convert MLX fp32 | 2026-05-15 00:19:09 CST | 2026-05-15 00:21:51 CST | 162 s |

Total staged first-run preparation time: 531 s.

## Download Output

```text
[xGRN] downloading GRN weights from HuggingFace: repo=bytedance-research/GRN revision=default attempt=1/2
Fetching 8 files: 100%|##########| 8/8 [04:50<00:00, 36.35s/it]
/tmp/xgrn-firststart-20260515-001232/GRN
```

The run used an empty model directory and an isolated `HF_HOME`; `HF_HOME`
remained tiny, so the downloaded files were materialized in the requested model
cache.

## Disk Usage

```text
58G   /tmp/xgrn-firststart-20260515-001232/GRN
9.5G  /tmp/xgrn-firststart-20260515-001232/GRN/mlx
19G   /tmp/xgrn-firststart-20260515-001232/GRN/mlx_fp32
968K  /tmp/xgrn-firststart-20260515-001232/hf-home
```

Large artifacts:

```text
8.2G  GRN_T2I_2B.pth
8.2G  GRN_T2V_2B.pth
2.6G  HBQ_tokenizer_64dim_M4.ckpt
11G   umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth
4.1G  mlx/grn_t2i_fp16.safetensors
4.1G  mlx/grn_t2v_fp16.safetensors
1.3G  mlx/hbq_fp16.safetensors
8.2G  mlx_fp32/grn_t2i_fp32.safetensors
8.2G  mlx_fp32/grn_t2v_fp32.safetensors
2.6G  mlx_fp32/hbq_fp32.safetensors
```

## App Validation

Bootstrap-only with the fresh cache:

```text
[xGRN] official GRN source ready: /Users/bytedance/code/GRN
[xGRN] raw model cache ready: /tmp/xgrn-firststart-20260515-001232/GRN
[xGRN] MLX artifacts ready: /tmp/xgrn-firststart-20260515-001232/GRN
[xGRN] bootstrap complete: /tmp/xgrn-firststart-20260515-001232/GRN
```

Full app launch on port 7861:

```text
[xGRN] official GRN source ready: /Users/bytedance/code/GRN
[xGRN] raw model cache ready: /tmp/xgrn-firststart-20260515-001232/GRN
[xGRN] MLX artifacts ready: /tmp/xgrn-firststart-20260515-001232/GRN
* Running on local URL:  http://127.0.0.1:7861
```

`curl -fsS http://127.0.0.1:7861/` returned the Gradio HTML document. The 7861
test server was stopped after validation; the existing 7860 process was not
touched.
