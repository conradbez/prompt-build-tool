/** API client for the pbt server (proxied through Vite at /api → localhost:8000) */

const BASE = '/api';

export interface DagNodePayload {
  name: string;
  source: string;
}

export interface DagResponse {
  dag_id: string;
}

export interface RunResponse {
  outputs: Record<string, string>;
  errors: string[];
}

/** Register a DAG from the editor and get back a stable dag_id. */
export async function submitDag(nodes: DagNodePayload[]): Promise<DagResponse> {
  const res = await fetch(`${BASE}/dag`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /dag failed (${res.status}): ${text}`);
  }
  return res.json();
}

/** Run one or more models from a previously registered DAG. */
export async function runDag(
  dagId: string,
  select?: string[],
  promptdata?: Record<string, string>,
): Promise<RunResponse> {
  const res = await fetch(`${BASE}/dag/${dagId}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ select, promptdata }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST /dag/${dagId}/run failed (${res.status}): ${text}`);
  }
  return res.json();
}
