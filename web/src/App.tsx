import { useEffect, useMemo, useRef, useState } from 'react'
import { Sparkles } from 'lucide-react'
import { Header, type ViewKey } from './components/Header'
import { PromptPanel } from './components/PromptPanel'
import { Controls, type ControlsState } from './components/Controls'
import { Advanced, defaultAdvanced, type AdvancedState } from './components/Advanced'
import { Preview } from './components/Preview'
import { Gallery } from './components/Gallery'
import { About } from './components/About'
import { Button, Toast } from './components/ui'
import { api, type GenerateResponse, type PresetMeta } from './lib/api'

export default function App() {
  const [view, setView] = useState<ViewKey>('create')
  const [presets, setPresets] = useState<PresetMeta | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [prompt, setPrompt] = useState('')
  const [controls, setControls] = useState<ControlsState>({
    task: 'T2I',
    quality: 'Balanced',
    aspect: '1:1 Square',
    seed: 42,
    duration: 0.5,
  })
  const [adv, setAdv] = useState<AdvancedState | null>(null)

  const [result, setResult] = useState<GenerateResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [statusMsg, setStatusMsg] = useState<string>('')
  const tickRef = useRef<number | null>(null)

  // Initial load: presets + last image
  useEffect(() => {
    api.presets()
      .then((p) => {
        setPresets(p)
        setPrompt(p.examples[0] ?? '')
        setAdv(defaultAdvanced(p))
      })
      .catch((e) => setError(`Backend not reachable: ${e.message}`))
  }, [])

  // Live elapsed timer while generating
  useEffect(() => {
    if (loading) {
      const start = performance.now()
      tickRef.current = window.setInterval(() => {
        setElapsed((performance.now() - start) / 1000)
      }, 100)
      return () => { if (tickRef.current) window.clearInterval(tickRef.current) }
    } else {
      setElapsed(0)
    }
  }, [loading])

  const initialImage = useMemo(() => api.fileUrl('outputs/latest_t2i.png') + `?_=${Date.now()}`, [])

  const generate = async () => {
    if (!presets || !adv) return
    setLoading(true)
    setStatusMsg('queued')
    setError(null)
    try {
      const res = await api.generate(
        {
          task: controls.task,
          prompt,
          quality: controls.quality,
          aspect: controls.aspect,
          seed: controls.seed,
          duration: controls.duration,
          ...adv,
        },
        (msg) => setStatusMsg(msg),
      )
      setResult(res)
    } catch (e: any) {
      setError(`Generation failed: ${e.message}`)
    } finally {
      setLoading(false)
      setStatusMsg('')
    }
  }

  // Cmd/Ctrl + Enter triggers generate from anywhere
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault()
        if (!loading && presets && adv) generate()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [loading, presets, adv, prompt, controls, view])

  const updateControls = (next: Partial<ControlsState>) => setControls((s) => ({ ...s, ...next }))
  const updateAdv = (next: Partial<AdvancedState>) => setAdv((s) => (s ? { ...s, ...next } : s))

  if (!presets || !adv) {
    return (
      <div className="min-h-screen grid place-items-center text-muted">
        <div className="flex items-center gap-3">
          <span className="spin h-5 w-5 rounded-full border-2 border-line border-t-terra" />
          <span className="text-sm">Loading…</span>
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-page px-5 sm:px-8 lg:px-10 py-8 lg:py-10 pb-20">
      <Header view={view} onView={setView} />

      {view === 'create' && (
        <main
          className="
            grid gap-6
            grid-cols-1
            lg:grid-cols-[minmax(380px,440px)_minmax(0,1fr)]
            xl:gap-8
            items-start
          "
        >
          <section className="space-y-4">
            <PromptPanel
              value={prompt}
              onChange={setPrompt}
              examples={presets.examples}
              onPick={setPrompt}
            />
            <Controls state={controls} setState={updateControls} presets={presets} />
            <div className="relative">
              <Button
                variant="primary"
                onClick={generate}
                loading={loading}
                className="w-full h-12 text-[0.95rem]"
              >
                <Sparkles className="h-4 w-4" />
                {loading ? 'Generating…' : 'Generate'}
              </Button>
              {!loading && (
                <span className="hidden sm:inline-flex absolute right-3 top-1/2 -translate-y-1/2 items-center gap-0.5 text-2xs text-white/70 pointer-events-none select-none">
                  <Kbd>⌘</Kbd><Kbd>↵</Kbd>
                </span>
              )}
            </div>
            <Advanced state={adv} setState={updateAdv} />
          </section>
          <section className="lg:sticky lg:top-6">
            <Preview
              result={result}
              loading={loading}
              elapsed={elapsed}
              statusMsg={statusMsg}
              initialImage={result ? null : initialImage}
            />
          </section>
        </main>
      )}

      {view === 'gallery' && <Gallery />}
      {view === 'about' && <About />}

      {error && <Toast message={error} onClose={() => setError(null)} />}
    </div>
  )
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex items-center justify-center min-w-[1.1rem] h-[1.1rem] px-1 rounded text-2xs font-mono bg-white/15 border border-white/20 leading-none">
      {children}
    </kbd>
  )
}
