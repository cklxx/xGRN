import { Card, Textarea } from './ui'

export function PromptPanel({
  value,
  onChange,
  examples,
  onPick,
}: {
  value: string
  onChange: (v: string) => void
  examples: string[]
  onPick: (v: string) => void
}) {
  return (
    <Card className="p-5">
      <label className="block text-2xs font-semibold uppercase tracking-cap text-muted mb-2">Prompt</label>
      <Textarea
        value={value}
        onChange={(e) => onChange(e.currentTarget.value)}
        rows={4}
        placeholder="A realistic photo of an orange tabby cat sitting on a windowsill…"
        className="min-h-[96px]"
      />
      {examples.length > 0 && (
        <div className="mt-4 -mb-1 flex flex-wrap gap-1.5">
          {examples.map((ex, i) => (
            <button
              key={i}
              onClick={() => onPick(ex)}
              title={ex}
              className="
                max-w-[260px] truncate
                px-3 py-1.5 rounded-full
                bg-sub border border-line text-xs text-ink
                hover:bg-card hover:border-terra/40 hover:text-ink
                transition-all
              "
            >
              {firstWords(ex, 7)}
            </button>
          ))}
        </div>
      )}
    </Card>
  )
}

function firstWords(s: string, n: number) {
  const parts = s.split(/\s+/)
  if (parts.length <= n) return s
  return parts.slice(0, n).join(' ') + '…'
}
