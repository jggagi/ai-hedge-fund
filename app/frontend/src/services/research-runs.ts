import { API_BASE_URL } from '@/lib/api-config';

export type RunStatus = 'IDLE' | 'IN_PROGRESS' | 'CANCEL_REQUESTED' | 'COMPLETE' | 'ERROR' | 'CANCELLED' | 'TIMED_OUT';
export interface ResearchRun {
  id: number;
  flow_id: number;
  run_number: number;
  status: RunStatus;
  request_data: Record<string, any> | null;
  results: Record<string, any> | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}
export const isRunActive = (status?: RunStatus) => status === 'IN_PROGRESS' || status === 'CANCEL_REQUESTED';
export const runStatusLabel: Record<RunStatus, string> = {
  IDLE: 'Ready', IN_PROGRESS: 'Running', CANCEL_REQUESTED: 'Waiting for execution to stop',
  COMPLETE: 'Complete', ERROR: 'Failed', CANCELLED: 'Stopped', TIMED_OUT: 'Timed out',
};

async function readResponse(response: Response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`);
  }
  return response.json();
}

export const researchRuns = {
  async list(flowId: number): Promise<ResearchRun[]> {
    return readResponse(await fetch(`${API_BASE_URL}/flows/${flowId}/runs/`));
  },
  async latest(flowId: number): Promise<ResearchRun | null> {
    const response = await fetch(`${API_BASE_URL}/flows/${flowId}/runs/latest`);
    if (response.status === 404) return null;
    return readResponse(response);
  },
  async get(flowId: number, runId: number): Promise<ResearchRun> {
    return readResponse(await fetch(`${API_BASE_URL}/flows/${flowId}/runs/${runId}`));
  },
  async cancel(flowId: number, runId: number): Promise<ResearchRun> {
    return readResponse(await fetch(`${API_BASE_URL}/flows/${flowId}/runs/${runId}/cancel`, { method: 'POST' }));
  },
};

export function exportResearchRun(run: ResearchRun) {
  const blob = new Blob([JSON.stringify({ schema_version: 1, run }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `research-${run.flow_id}-${run.id}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
