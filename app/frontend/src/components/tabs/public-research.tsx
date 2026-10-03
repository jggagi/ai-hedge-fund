import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useNodeContext } from '@/contexts/node-context';
import { useFlowConnection } from '@/hooks/use-flow-connection';
import { LanguageModel } from '@/data/models';
import { api } from '@/services/api';
import { flowService } from '@/services/flow-service';
import { ModelProvider } from '@/services/types';
import { Flow } from '@/types/flow';
import { ResearchHistory } from '@/components/panels/bottom/tabs/research-history';
import { ResearchRun } from '@/services/research-runs';

export function PublicResearch({ flow }: { flow: Flow }) {
  const [ticker, setTicker] = useState(flow.data?.ticker || 'AAPL');
  const [asOf, setAsOf] = useState(flow.data?.asOf || new Date().toISOString().slice(0, 10));
  const [modelName, setModelName] = useState(flow.data?.modelName || '');
  const [models, setModels] = useState<LanguageModel[]>([]);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [savedRun, setSavedRun] = useState<ResearchRun | null>(null);
  const { runFlow, stopFlow, canRun, isConnecting, isConnected, error: runError } = useFlowConnection(String(flow.id));
  const { getAgentNodeDataForFlow } = useNodeContext();
  useEffect(() => {
    let disposed = false;
    api.getLanguageModels().then(all => { if (!disposed) setModels(all.filter(model => model.provider === 'Ollama')); })
      .catch(e => { if (!disposed) setError(e instanceof Error ? e.message : 'Cannot check local models'); });
    return () => { disposed = true; };
  }, []);
  useEffect(() => {
    let disposed = false;
    flowService.getFlow(flow.id).then(latest => {
      if (disposed) return;
      if (latest.data?.ticker) setTicker(latest.data.ticker);
      if (latest.data?.asOf) setAsOf(latest.data.asOf);
      if (latest.data?.modelName) setModelName(latest.data.modelName);
    }).catch(() => {});
    return () => { disposed = true; };
  }, [flow.id]);
  const run = async () => {
    setError('');
    if (!models.some(model => model.model_name === modelName)) { setError('Choose an available local model.'); return; }
    if (!asOf || asOf > new Date().toISOString().slice(0, 10)) { setError('Choose a disclosure cutoff no later than today.'); return; }
    setSaving(true);
    try {
      await flowService.updateFlow(flow.id, { data: { ...flow.data, researchPreset: 'sec_filings', ticker, asOf, modelName } });
      setSavedRun(null);
      runFlow({ flow_id: flow.id, data_source: 'sec_filings', tickers: [ticker], graph_nodes: [], graph_edges: [],
        model_provider: ModelProvider.OLLAMA, model_name: modelName, end_date: asOf,
        start_date: `${Number(asOf.slice(0, 4)) - 2}${asOf.slice(4)}`, timeout_seconds: 300 });
    } catch (e) { setError(e instanceof Error ? e.message : 'Cannot save research configuration'); }
    finally { setSaving(false); }
  };
  const active = isConnecting || isConnected;
  const progress = Object.entries(getAgentNodeDataForFlow(String(flow.id)));
  return <div className="h-full overflow-y-auto p-4 md:p-6 space-y-5">
    <header><h1 className="text-xl font-semibold">Public filing research</h1>
      <p className="text-sm text-muted-foreground mt-2">Two local analysts review SEC annual disclosures: business fundamentals and risks. Live prices and valuation are outside this report.</p></header>
    <div className="flex flex-wrap gap-4 items-end rounded-lg border p-4">
      <label className="text-xs space-y-2">Company<select aria-label="Research company" className="block bg-background border rounded p-2" value={ticker} disabled={active} onChange={e => setTicker(e.target.value)}>
        <option>AAPL</option><option>MSFT</option><option>NVDA</option>
      </select></label>
      <label className="text-xs space-y-2">Disclosures filed by<Input aria-label="Disclosure cutoff" type="date" max={new Date().toISOString().slice(0, 10)} value={asOf} disabled={active} onChange={e => setAsOf(e.target.value)} /></label>
      <label className="text-xs space-y-2">Local model<select aria-label="Local research model" className="block bg-background border rounded p-2 max-w-full" value={modelName} disabled={active} onChange={e => setModelName(e.target.value)}>
        <option value="">Choose a local model</option>{models.map(model => <option key={model.model_name} value={model.model_name}>{model.model_name}</option>)}
      </select></label>
      <Button onClick={() => void run()} disabled={!canRun || saving || !modelName}>Run filing research</Button>
      {active && <Button variant="outline" onClick={() => void stopFlow()}>Stop research</Button>}
    </div>
    {(error || (!savedRun && runError)) && <p role="alert" className="text-sm text-destructive">{error || runError}</p>}
    <ResearchHistory flowId={flow.id} onSelect={setSavedRun} />
    {active && progress.length > 0 && <section aria-label="Research progress" className="text-xs space-y-2">
      {progress.map(([agent, data]) => <p key={agent}>{agent}: {data.message || data.status}</p>)}
    </section>}
    {savedRun?.status === 'COMPLETE' && <p className="text-xs text-muted-foreground">This saved report is available without another model call.</p>}
  </div>;
}
