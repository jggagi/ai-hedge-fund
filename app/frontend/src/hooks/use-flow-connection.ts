import { useNodeContext } from '@/contexts/node-context';
import { api } from '@/services/api';
import { backtestApi } from '@/services/backtest-api';
import { BacktestRequest, HedgeFundRequest } from '@/services/types';
import { isRunActive, ResearchRun, researchRuns } from '@/services/research-runs';
import { useCallback, useEffect, useRef, useState } from 'react';

// Connection state for a specific flow
export type FlowConnectionState = 'idle' | 'connecting' | 'connected' | 'error' | 'completed' | 'cancelling' | 'cancelled' | 'timed_out';

interface FlowConnectionInfo {
  state: FlowConnectionState;
  abortController: (() => void) | null;
  startTime: number;
  lastActivity: number;
  error?: string;
  runId?: number | null;
  run?: ResearchRun | null;
  kind?: 'single' | 'backtest';
  stopRequested?: boolean;
}

// Global connection manager - tracks all active flow connections
class FlowConnectionManager {
  private connections = new Map<string, FlowConnectionInfo>();
  private listeners = new Set<() => void>();

  // Get connection info for a flow
  getConnection(flowId: string): FlowConnectionInfo {
    return this.connections.get(flowId) || {
      state: 'idle',
      abortController: null,
      startTime: 0,
      lastActivity: 0,
    };
  }

  // Set connection info for a flow
  setConnection(flowId: string, info: Partial<FlowConnectionInfo>): void {
    const existing = this.getConnection(flowId);
    const updated = {
      ...existing,
      ...info,
      lastActivity: Date.now(),
    };
    
    this.connections.set(flowId, updated);
    this.notifyListeners();
  }

  // Remove connection for a flow
  removeConnection(flowId: string): void {
    const connection = this.connections.get(flowId);
    if (connection?.abortController) {
      connection.abortController();
    }
    this.connections.delete(flowId);
    this.notifyListeners();
  }

  // Add listener for connection changes
  addListener(listener: () => void): void {
    this.listeners.add(listener);
  }

  // Remove listener
  removeListener(listener: () => void): void {
    this.listeners.delete(listener);
  }

  // Notify all listeners of changes
  private notifyListeners(): void {
    this.listeners.forEach(listener => listener());
  }
}

// Global instance
export const flowConnectionManager = new FlowConnectionManager();

/**
 * Hook for managing flow connections and execution
 * @param flowId The ID of the flow to manage
 * @returns Connection state and control functions
 */
export function useFlowConnection(flowId: string | null) {
  const nodeContext = useNodeContext();
  const [, forceUpdate] = useState({});
  const listenerRef = useRef<() => void>();
  const contextRef = useRef(nodeContext);
  contextRef.current = nodeContext;

  // A stream is only a view of the durable run. Reloads retrieve it without rerunning.
  useEffect(() => {
    if (!flowId) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    let applied = '';
    const recover = async () => {
      try {
        const run = await researchRuns.latest(Number(flowId));
        if (disposed) return;
        const current = flowConnectionManager.getConnection(flowId);
        if (current.kind === 'backtest' && ['connecting', 'connected'].includes(current.state)) return;
        const waitingForStart = ['connecting', 'connected'].includes(current.state) && !current.runId;
        if (run && !waitingForStart && (!current.runId || run.id >= current.runId)) {
          const active = isRunActive(run.status);
          const state: FlowConnectionState = run.status === 'COMPLETE' ? 'completed'
            : run.status === 'CANCEL_REQUESTED' ? 'cancelling'
            : run.status === 'CANCELLED' ? 'cancelled'
            : run.status === 'TIMED_OUT' ? 'timed_out'
            : run.status === 'ERROR' ? 'error' : active ? 'connected' : 'idle';
          flowConnectionManager.setConnection(flowId, { state, runId: run.id, run, error: run.error_message || undefined });
          const signature = `${run.id}:${run.status}`;
          if (!active && applied !== signature) {
            applied = signature;
            const context = contextRef.current;
            context.resetNodeStatuses(flowId);
            if (run.results) context.setOutputNodeData(flowId, run.results as any);
          }
        }
      } catch (error) {
        // Keep the last known execution state. Losing API connectivity is not completion.
        if (!disposed) flowConnectionManager.setConnection(flowId, { error: error instanceof Error ? error.message : 'Cannot recover run' });
      } finally {
        if (!disposed) timer = setTimeout(recover, 2500);
      }
    };
    void recover();
    return () => { disposed = true; clearTimeout(timer); };
  }, [flowId]);

  // Force re-render when connections change
  useEffect(() => {
    const listener = () => forceUpdate({});
    listenerRef.current = listener;
    flowConnectionManager.addListener(listener);
    
    return () => {
      if (listenerRef.current) {
        flowConnectionManager.removeListener(listenerRef.current);
      }
    };
  }, []);

  // Get current connection state
  const connection = flowId ? flowConnectionManager.getConnection(flowId) : null;
  const isConnecting = connection?.state === 'connecting';
  const isConnected = connection?.state === 'connected' || connection?.state === 'cancelling';
  const isError = connection?.state === 'error';
  const isCompleted = connection?.state === 'completed';
  
  // Check if any agents are currently processing
  const isProcessing = flowId ? (() => {
    const agentData = nodeContext.getAgentNodeDataForFlow(flowId);
    return Object.values(agentData).some(agent => agent.status === 'IN_PROGRESS');
  })() : false;
  
  // Can run if we have a flow ID and we're not already running
  const canRun = Boolean(flowId && !isConnecting && !isConnected && !isProcessing);

  // Start a flow connection
  const runFlow = useCallback((params: HedgeFundRequest) => {
    if (!flowId || !canRun) return;

    // Reset node states for this flow
    nodeContext.resetAllNodes(flowId);

    // Set connecting state
    flowConnectionManager.setConnection(flowId, {
      state: 'connecting',
      startTime: Date.now(),
      runId: null,
      run: null,
      error: undefined,
      kind: 'single',
      stopRequested: false,
    });

    try {
      // Start the API call
      const abortController = api.runHedgeFund({ ...params, flow_id: Number(flowId) }, nodeContext, flowId);

      // Update connection with abort controller
      flowConnectionManager.setConnection(flowId, {
        state: 'connected',
        abortController,
      });

      // TODO: We should enhance the API to notify us when the connection completes
      // For now, we'll rely on the complete event from the SSE stream
      
    } catch (error) {
      console.error('Failed to start hedge fund run:', error);
      flowConnectionManager.setConnection(flowId, {
        state: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        abortController: null,
      });
    }
  }, [flowId, canRun, nodeContext]);

  // Start a backtest connection
  const runBacktest = useCallback((params: BacktestRequest) => {
    if (!flowId || !canRun) return;

    // Reset node states for this flow
    nodeContext.resetAllNodes(flowId);

    // Set connecting state
    flowConnectionManager.setConnection(flowId, {
      state: 'connecting',
      startTime: Date.now(),
      kind: 'backtest',
      runId: null,
      run: null,
    });

    try {
      // Start the backtest API call
      const abortController = backtestApi.runBacktest(params, nodeContext, flowId);

      // Update connection with abort controller
      flowConnectionManager.setConnection(flowId, {
        state: 'connected',
        abortController,
      });

      // TODO: We should enhance the API to notify us when the connection completes
      // For now, we'll rely on the complete event from the SSE stream
      
    } catch (error) {
      console.error('Failed to start backtest:', error);
      flowConnectionManager.setConnection(flowId, {
        state: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        abortController: null,
      });
    }
  }, [flowId, canRun, nodeContext]);

  // Stop a flow connection
  const stopFlow = useCallback(async () => {
    if (!flowId) return;
    const connection = flowConnectionManager.getConnection(flowId);
    if (connection.kind !== 'backtest' && connection.runId) {
      try {
        const run = await researchRuns.cancel(Number(flowId), connection.runId);
        flowConnectionManager.setConnection(flowId, {
          state: isRunActive(run.status) ? 'cancelling' : run.status === 'COMPLETE' ? 'completed' : 'cancelled',
          run,
        });
      } catch (error) {
        flowConnectionManager.setConnection(flowId, { error: error instanceof Error ? error.message : 'Cancellation failed' });
      }
      return;
    }
    // Before the start event arrives a saved run may already exist. Recover its ID.
    if (connection.kind === 'single' && ['connecting', 'connected'].includes(connection.state)) {
      flowConnectionManager.setConnection(flowId, { stopRequested: true });
      return;
    }
    
    if (connection.abortController) {
      connection.abortController();
    }

    // Reset only node statuses when stopping, preserving all data (backtest results, messages, etc.)
    nodeContext.resetNodeStatuses(flowId);

    // Update connection state
    flowConnectionManager.setConnection(flowId, {
      state: 'idle',
      abortController: null,
    });
    
  }, [flowId, nodeContext]);

  // Recover from stale states (called when loading a flow)
  const recoverFlowState = useCallback(() => {
    if (!flowId) return;

    const connection = flowConnectionManager.getConnection(flowId);
    if (connection.runId) return;
    
    // If we think we're connected but have no processing nodes, we're probably stale
    if ((connection.state === 'connected' || connection.state === 'connecting') && !isProcessing) {
      // Check if the connection is old (more than 5 minutes)
      const isStale = Date.now() - connection.lastActivity > 5 * 60 * 1000;
      
      if (isStale) {
        console.log(`Recovering stale connection for flow ${flowId}`);
        flowConnectionManager.setConnection(flowId, {
          state: 'idle',
          abortController: null,
        });
      }
    }
  }, [flowId, isProcessing]);

  return {
    // State
    isConnecting,
    isConnected,
    isError,
    isCompleted,
    isProcessing,
    canRun,
    error: connection?.error,
    
    // Actions
    runFlow,
    runBacktest,
    stopFlow,
    recoverFlowState,
  };
}

// Utility hook to get connection state for any flow (for monitoring)
export function useFlowConnectionState(flowId: string | null) {
  const [, forceUpdate] = useState({});

  useEffect(() => {
    if (!flowId) return;

    const unsubscribe = flowConnectionManager.addListener(() => {
      forceUpdate({});
    });

    return unsubscribe;
  }, [flowId]);

  return flowId ? flowConnectionManager.getConnection(flowId) : null;
}
