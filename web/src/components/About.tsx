import { Card } from './ui'

export function About() {
  return (
    <div className="prose-narrow space-y-6 text-ink">
      <Card className="p-7">
        <h3 className="text-lg font-semibold mb-2">What this is</h3>
        <p className="text-sm leading-relaxed text-ink">
          A Mac-specialised runtime for the official <b>GRN</b> T2I/T2V models. Text encoding is bf16 UMT5,
          the GRN transformer + refinement loop run in <b>MLX</b>, and HBQ decode uses a native MLX decoder.
          Up to <b>102×</b> faster than stock PyTorch/MPS on the same Mac.
        </p>
      </Card>

      <Card className="p-7">
        <h3 className="text-lg font-semibold mb-3">How to use</h3>
        <ol className="list-decimal pl-5 text-sm leading-relaxed space-y-1">
          <li>Type a prompt (or pick an example).</li>
          <li>Choose <b>Image</b> or <b>Video</b>, pick a <b>Quality</b> preset, choose an aspect.</li>
          <li>Hit <b>Generate</b>.</li>
        </ol>
        <p className="mt-3 text-sm text-muted">
          Power-user knobs (dtypes, kernel fusion, sampling mode) live under <b>Advanced</b>.
        </p>
      </Card>

      <Card className="p-7">
        <h3 className="text-lg font-semibold mb-2">First-run download</h3>
        <p className="text-sm leading-relaxed">
          The first time you hit <b>Generate</b>, xGRN downloads the official GRN HuggingFace snapshot
          into <Code>models/GRN/</Code> and creates MLX artifacts. Plan for several GB free disk and a
          few minutes on a fast network. Subsequent launches reuse the cache.
        </p>
        <p className="mt-3 text-sm text-muted">You can also pre-fetch from the terminal:</p>
        <pre className="mt-2 rounded-md bg-pre text-bg px-4 py-3 text-xs font-mono overflow-x-auto">{`uv run xgrn-download --model-dir models/GRN`}</pre>
      </Card>

      <Card className="p-7">
        <h3 className="text-lg font-semibold mb-2">Outputs</h3>
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-line">
              <th className="text-left py-2 font-semibold text-muted">File</th>
              <th className="text-left py-2 font-semibold text-muted">Description</th>
            </tr>
          </thead>
          <tbody className="text-ink">
            <Row file="outputs/latest_t2i.png" desc="Most recent image" />
            <Row file="outputs/latest_t2v.mp4" desc="Most recent video" />
            <Row file="outputs/latest_t2v_first_frame.png" desc="First-frame preview" />
            <Row file="outputs/refinement_stats.csv" desc="Per-step metrics" />
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function Row({ file, desc }: { file: string; desc: string }) {
  return (
    <tr className="border-b border-line/60 last:border-0">
      <td className="py-2 pr-4"><Code>{file}</Code></td>
      <td className="py-2 text-muted">{desc}</td>
    </tr>
  )
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="font-mono text-xs px-1.5 py-0.5 rounded bg-sub border border-line text-terra">
      {children}
    </code>
  )
}
