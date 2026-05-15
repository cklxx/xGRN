import { Dice5, Image as ImageIcon, Video, Zap, Diamond, Sparkles } from 'lucide-react'
import { Card, Segmented, FieldLabel, NativeSelect, Input, Button, Slider } from './ui'
import type { Task, QualityKey, PresetMeta } from '../lib/api'
import { randomSeed } from '../lib/api'

export interface ControlsState {
  task: Task
  quality: QualityKey
  aspect: string
  seed: number
  duration: number
}

export function Controls({
  state,
  setState,
  presets,
}: {
  state: ControlsState
  setState: (next: Partial<ControlsState>) => void
  presets: PresetMeta
}) {
  const aspectKeys = Object.keys(presets.aspects)
  return (
    <Card className="p-5 space-y-4">
      <div>
        <FieldLabel>Mode</FieldLabel>
        <Segmented<Task>
          value={state.task}
          onChange={(v) => setState({ task: v })}
          items={[
            { value: 'T2I', label: <><ImageIcon className="h-4 w-4" /> Image</> },
            { value: 'T2V', label: <><Video className="h-4 w-4" /> Video</> },
          ]}
        />
      </div>

      <div>
        <FieldLabel hint={presets.qualities[state.quality]?.hint}>Quality preset</FieldLabel>
        <Segmented<QualityKey>
          value={state.quality}
          onChange={(v) => setState({ quality: v })}
          items={[
            { value: 'Fast', label: <><Zap className="h-3.5 w-3.5" /> Fast</> },
            { value: 'Balanced', label: <><Diamond className="h-3.5 w-3.5" /> Balanced</> },
            { value: 'Quality', label: <><Sparkles className="h-3.5 w-3.5" /> Quality</> },
          ]}
        />
      </div>

      <div className="grid grid-cols-[1fr_120px_auto] gap-2.5 items-end">
        <div>
          <FieldLabel>Aspect</FieldLabel>
          <NativeSelect
            value={state.aspect}
            onChange={(e) => setState({ aspect: e.currentTarget.value })}
          >
            {aspectKeys.map((k) => <option key={k} value={k}>{k}</option>)}
          </NativeSelect>
        </div>
        <div>
          <FieldLabel>Seed</FieldLabel>
          <Input
            type="number"
            value={state.seed}
            onChange={(e) => setState({ seed: Number(e.currentTarget.value) || 0 })}
            className="font-mono text-sm"
          />
        </div>
        <div>
          <FieldLabel>&nbsp;</FieldLabel>
          <Button
            variant="icon"
            onClick={() => setState({ seed: randomSeed() })}
            title="Random seed"
            aria-label="Random seed"
            className="h-10 w-10 px-0"
          >
            <Dice5 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {state.task === 'T2V' && (
        <div className="pt-1">
          <Slider
            label="Video duration"
            value={state.duration}
            min={0.25}
            max={2}
            step={0.25}
            onChange={(v) => setState({ duration: v })}
            format={(v) => `${v.toFixed(2)}s`}
          />
        </div>
      )}
    </Card>
  )
}
