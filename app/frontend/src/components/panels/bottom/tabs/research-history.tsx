import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { exportResearchRun, isRunActive, ResearchRun, researchRuns, runStatusLabel } from '@/services/research-runs';

export function ResearchHistory({ flowId, onSelect }: { flowId: number; onSelect: (run: ResearchRun | null) => void }) {
  const [runs, setRuns] = useState<ResearchRun[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selected, setSelected] = useState<ResearchRun | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let disposed = false;
    const load = async () => {
      try {
        const records = await researchRuns.list(flowId);
        if (disposed) return;
        setRuns(records);
        setError('');
      } catch (e) { if (!disposed) setError(e instanceof Error ? e.message : 'Cannot load history'); }
    };
    setSelectedId(null); setSelected(null); onSelect(null);
    void load();
    const timer = setInterval(load, 3000);
    return () => { disposed = true; clearInterval(timer); };
  }, [flowId, onSelect]);

  useEffect(() => {
    const id = selectedId ?? runs[0]?.id;
    if (!id) return;
    let disposed = false;
    researchRuns.get(flowId, id).then(run => {
      if (!disposed) { setSelected(run); onSelect(run); }
    }).catch(e => { if (!disposed) setError(e instanceof Error ? e.message : 'Cannot load run'); });
    return () => { disposed = true; };
  }, [flowId, runs, selectedId, onSelect]);

  const report = selected?.results?.research_report;
  const provenance = report?.provenance || report;
  const sources = provenance?.source_snapshots || report?.sources || [];
  return <section className="mb-4 rounded-lg border p-4 space-y-3" aria-label="Research history">
    <div className="flex flex-wrap items-center gap-3">
      <h3 className="font-semibold">Research history</h3>
      <label className="flex flex-wrap items-center gap-2 text-xs">Saved run
        <select aria-label="Saved research run" className="bg-background border rounded p-2 max-w-full" value={selectedId ?? runs[0]?.id ?? ''}
          onChange={e => setSelectedId(Number(e.target.value))}>
          {!runs.length && <option value="">No saved runs yet</option>}
          {runs.map(run => <option key={run.id} value={run.id}>#{run.run_number} · {runStatusLabel[run.status]} · {run.created_at}</option>)}
        </select>
      </label>
      {selected && <Button size="sm" variant="outline" onClick={() => exportResearchRun(selected)}>Export saved run</Button>}
    </div>
    {error && <p role="alert" className="text-destructive text-xs">{error}</p>}
    {selected && <>
      <p role="status" className="text-xs">Run #{selected.id}: {runStatusLabel[selected.status]}
        {isRunActive(selected.status) && ' · You can reload this page; execution stays on the server.'}</p>
      {selected.error_message && <p className="text-xs text-destructive">{selected.error_message}</p>}
      {report && <>
        <p className="text-xs text-muted-foreground">Saved inputs are source observations. Analyst signals and decisions are model interpretations.</p>
        {report.interpretations?.map((note: any) => <article key={note.role} className="rounded border p-3 space-y-2">
          <h4 className="font-semibold">{note.role === 'fundamentals_analyst' ? 'Business fundamentals' : 'Disclosure risks'}</h4>
          <p className="text-sm whitespace-pre-wrap">{note.reasoning}</p>
          <p className="text-xs text-muted-foreground break-all">References: {note.cited_fact_ids?.join(', ')}</p>
        </article>)}
        <details><summary className="cursor-pointer">Model and data window</summary>
          <pre className="text-xs whitespace-pre-wrap break-all mt-2">{JSON.stringify({ data_window: provenance?.data_window, model_config: report.model_config, model_invocations: provenance?.model_invocations }, null, 2)}</pre>
        </details>
        {report.observations?.length > 0 && <div className="overflow-x-auto"><table className="text-xs w-full"><caption className="text-left font-semibold mb-2">Source observations</caption><thead><tr><th className="text-left p-2">Fact</th><th className="text-left p-2">Value</th><th className="text-left p-2">Period end</th><th className="text-left p-2">Filed</th></tr></thead><tbody>
          {report.observations.map((fact: any) => <tr key={fact.fact_id} className="border-t"><td className="p-2">{fact.label || fact.metric}</td><td className="p-2">{Number(fact.value).toLocaleString()} {fact.unit}</td><td className="p-2 whitespace-nowrap">{fact.end}</td><td className="p-2 whitespace-nowrap">{fact.filed}</td></tr>)}
        </tbody></table></div>}
        <details><summary className="cursor-pointer">Data sources and saved observations ({sources.length})</summary>
          <pre className="text-xs whitespace-pre-wrap break-all mt-2">{JSON.stringify(sources, null, 2)}</pre>
        </details>
        {(report.gaps || report.data_gaps)?.length > 0 && <div className="text-xs"><strong>Data gaps</strong><ul className="list-disc pl-5">{(report.gaps || report.data_gaps).map((gap: any, i: number) => <li key={i}>{typeof gap === 'string' ? gap : gap.reason || gap.error || JSON.stringify(gap)}</li>)}</ul></div>}
      </>}
    </>}
  </section>;
}
