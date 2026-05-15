import { Tabs } from './ui'

export type ViewKey = 'create' | 'gallery' | 'about'

export function Header({ view, onView }: { view: ViewKey; onView: (v: ViewKey) => void }) {
  return (
    <header className="mb-7">
      <div className="flex items-center justify-between gap-6 flex-wrap">
        <div className="flex items-center gap-3.5">
          <BrandMark />
          <div>
            <h1 className="text-[1.7rem] font-semibold tracking-[-0.025em] leading-none text-ink">xGRN</h1>
            <p className="mt-1 text-sm text-muted">GRN T2I · T2V on Apple Metal · native MLX runtime</p>
          </div>
        </div>
        <div className="hidden md:flex items-center gap-2">
          <Stat value="102×" label="vs PyTorch/MPS warm" />
          <Stat value="25×" label="cold start" tone="teal" />
          <Stat value="bf16" label="UMT5 + MLX" mono />
        </div>
      </div>

      <div className="mt-7">
        <Tabs
          value={view}
          onChange={onView}
          items={[
            { value: 'create', label: 'Create' },
            { value: 'gallery', label: 'Gallery' },
            { value: 'about', label: 'About' },
          ]}
        />
      </div>
    </header>
  )
}

function BrandMark() {
  return (
    <div
      className="grid place-items-center h-11 w-11 rounded-[10px] bg-terra text-white font-mono font-semibold text-[1.2rem] leading-none"
      style={{ boxShadow: '0 1px 0 rgba(255,255,255,0.4) inset, 0 2px 6px rgba(201,100,66,0.22)' }}
    >
      x
    </div>
  )
}

function Stat({ value, label, tone, mono }: { value: string; label: string; tone?: 'terra' | 'teal'; mono?: boolean }) {
  const valueColor = tone === 'teal' ? 'text-teal-600' : tone === undefined ? 'text-terra' : 'text-terra'
  return (
    <div className="flex items-baseline gap-2 px-3 py-1.5 rounded-md border border-line bg-sub">
      <span className={`font-mono font-semibold text-[0.85rem] ${valueColor}`}>{value}</span>
      <span className={`text-2xs text-muted ${mono ? 'font-mono' : ''}`}>{label}</span>
    </div>
  )
}
