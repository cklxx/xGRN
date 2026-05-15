// xGRN backend API. Base URL is empty so requests go through Vite's proxy in
// dev (vite.config proxies /api → 7860) and the same-origin FastAPI in prod.

export type Task = 'T2I' | 'T2V'
export type QualityKey = 'Fast' | 'Balanced' | 'Quality'

export interface GenerateRequest {
  task: Task
  prompt: string
  negative_prompt?: string
  seed: number
  quality: QualityKey
  aspect: string // ratio key, e.g. '1:1 Square'
  duration?: number // T2V seconds
  use_preset?: boolean
  custom_steps?: number
  custom_guidance?: number
  temperature?: number
  text_dtype?: string
  text_cache_dtype?: string
  weights_dtype?: string
  compute_dtype?: string
  decoder_backend?: string
  sampling_mode?: string
  mask_schedule?: string
  fuse_mlp_gate_up?: boolean
  fuse_swiglu_metal?: boolean
  stack_cfg_cache?: boolean
  min_change_frac?: number
  track_token_confidence?: boolean
  precompute_pt_embed?: boolean
  capture_interval?: number
}

export interface RunSummary {
  task: Task
  preset: QualityKey
  pn: string
  steps: number
  guidance: number
  elapsed_sec: number
  timings: Record<string, number>
  raw_shape: number[]
  output: string
}

export interface GenerateResponse {
  ok: boolean
  task: Task
  elapsed_sec: number
  image_url?: string | null
  video_url?: string | null
  caption: string
  summary: RunSummary
  stats: Array<Record<string, number | string>>
  refinement_frames: string[]
}

export interface HistoryItem {
  url: string
  filename: string
  mtime: number
  type: 'image' | 'video' | 'gif'
  size: number
}

export interface PresetMeta {
  qualities: Record<QualityKey, { pn: string; steps: number; guidance: number; label: string; hint: string }>
  aspects: Record<string, number>
  examples: string[]
  defaults: {
    negative_prompt: string
    text_dtype: string
    text_cache_dtype: string
    weights_dtype: string
    compute_dtype: string
    decoder_backend: string
    sampling_mode: string
    mask_schedule: string
    temperature: number
  }
  status: { model_dir: string; ready: boolean }
}

async function jsonOrThrow<T>(p: Promise<Response>): Promise<T> {
  const r = await p
  if (!r.ok) {
    const text = await r.text().catch(() => r.statusText)
    throw new Error(text || `HTTP ${r.status}`)
  }
  return r.json() as Promise<T>
}

export const api = {
  presets: () => jsonOrThrow<PresetMeta>(fetch('/api/presets')),
  history: () => jsonOrThrow<HistoryItem[]>(fetch('/api/history')),
  generate: (body: GenerateRequest) =>
    jsonOrThrow<GenerateResponse>(
      fetch('/api/generate', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      }),
    ),
  fileUrl: (path: string) => `/api/file/${path.replace(/^\/+/, '')}`,
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 2_147_483_647)
}
