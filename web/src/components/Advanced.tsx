import { Disclosure, FieldLabel, NativeSelect, Slider, Checkbox, Textarea } from './ui'
import type { PresetMeta } from '../lib/api'

export interface AdvancedState {
  use_preset: boolean
  custom_steps: number
  custom_guidance: number
  temperature: number
  negative_prompt: string
  text_dtype: string
  text_cache_dtype: string
  weights_dtype: string
  compute_dtype: string
  decoder_backend: string
  sampling_mode: string
  mask_schedule: string
  fuse_mlp_gate_up: boolean
  fuse_swiglu_metal: boolean
  stack_cfg_cache: boolean
  min_change_frac: number
  track_token_confidence: boolean
  precompute_pt_embed: boolean
  capture_interval: number
}

export const defaultAdvanced = (presets: PresetMeta): AdvancedState => ({
  use_preset: true,
  custom_steps: 50,
  custom_guidance: 3.0,
  temperature: presets.defaults.temperature,
  negative_prompt: presets.defaults.negative_prompt,
  text_dtype: presets.defaults.text_dtype,
  text_cache_dtype: presets.defaults.text_cache_dtype,
  weights_dtype: presets.defaults.weights_dtype,
  compute_dtype: presets.defaults.compute_dtype,
  decoder_backend: presets.defaults.decoder_backend,
  sampling_mode: presets.defaults.sampling_mode,
  mask_schedule: presets.defaults.mask_schedule,
  fuse_mlp_gate_up: false,
  fuse_swiglu_metal: false,
  stack_cfg_cache: false,
  min_change_frac: 0,
  track_token_confidence: false,
  precompute_pt_embed: false,
  capture_interval: 0,
})

export function Advanced({
  state,
  setState,
}: {
  state: AdvancedState
  setState: (next: Partial<AdvancedState>) => void
}) {
  return (
    <Disclosure title="Advanced — engineer mode">
      <div className="space-y-5">
        <div className="rounded-md border border-teal-200 bg-teal-50 px-3.5 py-2.5 text-xs text-teal-700">
          These knobs override the <b>Quality</b> preset and are only useful if you know what you're doing.
          Defaults are correctness-tested.
        </div>

        <Checkbox
          checked={state.use_preset}
          onChange={(v) => setState({ use_preset: v })}
          label="Use preset for steps & guidance"
          hint="Uncheck to override below"
        />

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Slider
            label="Refinement steps"
            value={state.custom_steps}
            min={4} max={250} step={1}
            onChange={(v) => setState({ custom_steps: v })}
          />
          <Slider
            label="Guidance"
            value={state.custom_guidance}
            min={1} max={7} step={0.1}
            onChange={(v) => setState({ custom_guidance: v })}
            format={(v) => v.toFixed(1)}
          />
          <Slider
            label="Temperature"
            value={state.temperature}
            min={0.6} max={1.6} step={0.05}
            onChange={(v) => setState({ temperature: v })}
            format={(v) => v.toFixed(2)}
          />
        </div>

        <div>
          <FieldLabel>Negative prompt</FieldLabel>
          <div className="rounded-md border border-line bg-card px-3 py-2.5">
            <Textarea
              value={state.negative_prompt}
              rows={2}
              onChange={(e) => setState({ negative_prompt: e.currentTarget.value })}
              className="text-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <SmallSelect label="Text dtype" value={state.text_dtype} onChange={(v) => setState({ text_dtype: v })} options={['bf16','fp16','fp32']} />
          <SmallSelect label="Text cache" value={state.text_cache_dtype} onChange={(v) => setState({ text_cache_dtype: v })} options={['fp32','fp16']} />
          <SmallSelect label="GRN weights" value={state.weights_dtype} onChange={(v) => setState({ weights_dtype: v })} options={['fp32','fp16','auto']} />
          <SmallSelect label="GRN compute" value={state.compute_dtype} onChange={(v) => setState({ compute_dtype: v })} options={['bf16','fp32','fp16']} />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <SmallSelect label="HBQ decoder" value={state.decoder_backend} onChange={(v) => setState({ decoder_backend: v })} options={['native','mps']} />
          <SmallSelect label="Sampling" value={state.sampling_mode} onChange={(v) => setState({ sampling_mode: v })} options={['categorical','binary','argmax']} />
          <SmallSelect label="Schedule" value={state.mask_schedule} onChange={(v) => setState({ mask_schedule: v })} options={['random','dus']} />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Checkbox checked={state.fuse_mlp_gate_up} onChange={(v) => setState({ fuse_mlp_gate_up: v })} label="Fuse MLP gate/up" />
          <Checkbox checked={state.fuse_swiglu_metal} onChange={(v) => setState({ fuse_swiglu_metal: v })} label="Fuse SwiGLU Metal" />
          <Checkbox checked={state.stack_cfg_cache} onChange={(v) => setState({ stack_cfg_cache: v })} label="Stack CFG cache" />
          <Checkbox checked={state.track_token_confidence} onChange={(v) => setState({ track_token_confidence: v })} label="Track confidence" />
          <Checkbox checked={state.precompute_pt_embed} onChange={(v) => setState({ precompute_pt_embed: v })} label="Precompute pt embed" />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Slider
            label="Early-stop threshold"
            value={state.min_change_frac}
            min={0} max={0.05} step={0.001}
            onChange={(v) => setState({ min_change_frac: v })}
            format={(v) => v.toFixed(3)}
          />
          <Slider
            label="Capture every N steps"
            value={state.capture_interval}
            min={0} max={25} step={1}
            onChange={(v) => setState({ capture_interval: v })}
            format={(v) => v === 0 ? 'off' : String(v)}
          />
        </div>
      </div>
    </Disclosure>
  )
}

function SmallSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: string[]
}) {
  return (
    <div>
      <FieldLabel>{label}</FieldLabel>
      <NativeSelect value={value} onChange={(e) => onChange(e.currentTarget.value)}>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </NativeSelect>
    </div>
  )
}
