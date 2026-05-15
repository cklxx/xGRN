import { useState, useRef, useEffect, type ReactNode, type ButtonHTMLAttributes, type InputHTMLAttributes, type TextareaHTMLAttributes, type SelectHTMLAttributes } from 'react'
import { ChevronDown, Check } from 'lucide-react'
import { cn } from '../lib/cn'

/* ─── Button ─────────────────────────────────────────────────────────── */

type Variant = 'primary' | 'secondary' | 'ghost' | 'icon'

const variantStyles: Record<Variant, string> = {
  primary:
    'bg-terra text-white shadow-terra hover:bg-terra-hover hover:shadow-terraHover hover:-translate-y-px active:translate-y-0',
  secondary:
    'bg-[#ECEAE4] text-ink border border-line hover:bg-[#E0DCD2]',
  ghost: 'bg-transparent text-ink hover:bg-[#ECEAE4]',
  icon: 'bg-[#ECEAE4] text-ink border border-line hover:bg-[#E0DCD2] aspect-square',
}

export function Button({
  variant = 'secondary',
  className,
  children,
  loading,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; loading?: boolean }) {
  return (
    <button
      {...props}
      disabled={loading || props.disabled}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-md px-4 h-10 text-sm font-medium transition-all',
        'disabled:opacity-60 disabled:cursor-not-allowed',
        'focus-visible:outline-none focus-visible:ring focus-visible:ring-terra/20',
        variantStyles[variant],
        className,
      )}
    >
      {loading && (
        <span className="spin inline-block h-4 w-4 rounded-full border-2 border-white/40 border-t-white" />
      )}
      {children}
    </button>
  )
}

/* ─── Card / Section / Label ─────────────────────────────────────────── */

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div
      className={cn(
        'rounded-xl bg-card border border-line shadow-soft',
        className,
      )}
    >
      {children}
    </div>
  )
}

export function FieldLabel({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <div className="mb-2 flex items-baseline justify-between gap-2">
      <label className="text-2xs font-semibold uppercase tracking-cap text-muted">{children}</label>
      {hint && <span className="text-2xs text-soft">{hint}</span>}
    </div>
  )
}

/* ─── Textarea ───────────────────────────────────────────────────────── */

export function Textarea({ className, ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={cn(
        'w-full bg-transparent text-ink placeholder:text-soft',
        'border-0 outline-none resize-none p-0',
        'text-base leading-relaxed',
        className,
      )}
    />
  )
}

/* ─── Input (text/number) ────────────────────────────────────────────── */

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        'w-full h-10 px-3 rounded-md bg-card text-ink',
        'border border-line outline-none',
        'transition-shadow focus:border-terra focus:shadow-ring',
        'placeholder:text-soft',
        className,
      )}
    />
  )
}

/* ─── Select (native, styled) ────────────────────────────────────────── */

export function NativeSelect({ className, children, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className={cn('relative', className)}>
      <select
        {...props}
        className={cn(
          'w-full h-10 pl-3 pr-9 rounded-md bg-card text-ink appearance-none',
          'border border-line outline-none cursor-pointer',
          'transition-shadow focus:border-terra focus:shadow-ring',
        )}
      >
        {children}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted" />
    </div>
  )
}

/* ─── Segmented control (radio) ──────────────────────────────────────── */

export interface SegmentItem<T extends string> {
  value: T
  label: ReactNode
  hint?: string
}

export function Segmented<T extends string>({
  items,
  value,
  onChange,
  className,
}: {
  items: SegmentItem<T>[]
  value: T
  onChange: (v: T) => void
  className?: string
}) {
  return (
    <div
      role="radiogroup"
      className={cn(
        'inline-flex w-full p-1 gap-1 rounded-md bg-[#ECEAE4]',
        className,
      )}
    >
      {items.map((it) => {
        const selected = it.value === value
        return (
          <button
            key={it.value}
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(it.value)}
            className={cn(
              'flex-1 inline-flex items-center justify-center gap-1.5 rounded',
              'h-9 px-3 text-sm font-medium transition-all',
              'focus-visible:outline-none',
              selected
                ? 'bg-card text-ink shadow-[0_1px_2px_rgba(28,25,23,0.06),0_0_0_1px_rgba(201,100,66,0.25)]'
                : 'text-muted hover:bg-white/60 hover:text-ink',
            )}
          >
            {it.label}
          </button>
        )
      })}
    </div>
  )
}

/* ─── Checkbox ───────────────────────────────────────────────────────── */

export function Checkbox({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: ReactNode
  hint?: ReactNode
}) {
  return (
    <label className="inline-flex items-start gap-2.5 cursor-pointer select-none group">
      <span
        className={cn(
          'mt-0.5 inline-flex h-4 w-4 items-center justify-center rounded border transition-all',
          checked ? 'bg-terra border-terra text-white' : 'bg-card border-line group-hover:border-muted',
        )}
      >
        {checked && <Check className="h-3 w-3 stroke-[3]" />}
      </span>
      <span className="text-sm leading-tight">
        <span className="text-ink">{label}</span>
        {hint && <span className="block text-2xs text-muted mt-0.5">{hint}</span>}
      </span>
      <input
        type="checkbox"
        className="sr-only"
        checked={checked}
        onChange={(e) => onChange(e.currentTarget.checked)}
      />
    </label>
  )
}

/* ─── Slider with value readout ──────────────────────────────────────── */

export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  format,
}: {
  label: ReactNode
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step?: number
  format?: (v: number) => string
}) {
  const display = format ? format(value) : String(value)
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <label className="text-2xs font-semibold uppercase tracking-cap text-muted">{label}</label>
        <span className="text-xs font-mono text-ink">{display}</span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.currentTarget.value))}
        className="w-full"
      />
    </div>
  )
}

/* ─── Disclosure (accordion) ─────────────────────────────────────────── */

export function Disclosure({
  title,
  defaultOpen = false,
  children,
  rightSlot,
}: {
  title: ReactNode
  defaultOpen?: boolean
  children: ReactNode
  rightSlot?: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-xl border border-line bg-card overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-5 py-3.5 text-left hover:bg-sub transition-colors"
      >
        <span className="text-sm font-medium text-ink">{title}</span>
        <span className="flex items-center gap-2">
          {rightSlot}
          <ChevronDown className={cn('h-4 w-4 text-muted transition-transform', open && 'rotate-180')} />
        </span>
      </button>
      {open && <div className="px-5 pb-5 pt-1 border-t border-line animate-fade-in">{children}</div>}
    </div>
  )
}

/* ─── Tabs (controlled) ──────────────────────────────────────────────── */

export function Tabs<T extends string>({
  value,
  onChange,
  items,
}: {
  value: T
  onChange: (v: T) => void
  items: { value: T; label: ReactNode }[]
}) {
  return (
    <div className="flex items-center gap-0 border-b border-line">
      {items.map((it) => {
        const selected = it.value === value
        return (
          <button
            key={it.value}
            onClick={() => onChange(it.value)}
            className={cn(
              'relative h-10 px-4 text-sm font-medium transition-colors',
              selected ? 'text-ink' : 'text-muted hover:text-ink',
            )}
          >
            {it.label}
            {selected && (
              <span className="absolute inset-x-3 -bottom-px h-0.5 bg-terra rounded-full" />
            )}
          </button>
        )
      })}
    </div>
  )
}

/* ─── Pill ───────────────────────────────────────────────────────────── */

export function Pill({
  children,
  tone = 'neutral',
  className,
}: {
  children: ReactNode
  tone?: 'neutral' | 'terra' | 'teal'
  className?: string
}) {
  const tones = {
    neutral: 'bg-[#ECEAE4] border-line text-muted',
    terra: 'bg-terra text-white border-terra',
    teal: 'bg-teal-100 text-teal-700 border-teal-200',
  }
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-2xs font-mono',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

/* ─── Toast (minimal, top-right) ─────────────────────────────────────── */

export function Toast({ message, onClose }: { message: string; onClose: () => void }) {
  useEffect(() => {
    const t = setTimeout(onClose, 6000)
    return () => clearTimeout(t)
  }, [onClose])
  return (
    <div
      role="alert"
      className="fixed top-6 right-6 z-50 max-w-md rounded-md bg-pre text-bg px-4 py-3 shadow-lg animate-fade-in"
    >
      <p className="text-sm">{message}</p>
    </div>
  )
}

/* ─── ScrollAnchor: tiny helper for scroll-into-view ─────────────────── */

export function useScrollIntoViewWhen(when: boolean) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (when && ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [when])
  return ref
}
