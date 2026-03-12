import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  type Node,
  type Edge,
  type Connection,
  type NodeChange,
  type OnConnectEnd,
  type ReactFlowInstance,
  BackgroundVariant,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useMutation } from '@tanstack/react-query';
import {
  PlusIcon, SendIcon, AlertCircleIcon, CheckCircleIcon,
  DatabaseIcon, FileIcon,
} from 'lucide-react';

import PromptNode, { type PromptNodeData } from './PromptNode';
import NodePanel from './NodePanel';
import PromptDataManager, { type PromptDataRow } from './PromptDataManager';
import PromptFileManager, { type PromptFileRow } from './PromptFileManager';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from '@/components/ui/dialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { submitDag, runDag } from '../api';

// ── localStorage persistence ──────────────────────────────────────────────────

const STORAGE_KEY = 'pbt_dag_state';

const _initialSavedState: Record<string, unknown> | null = (() => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
})();

// ── Module-level constants (stable across renders) ────────────────────────────

const nodeTypes = { promptNode: PromptNode };

// Stable no-op; only needed to satisfy ReactFlow's onConnectEnd prop type
const handleConnectEnd: OnConnectEnd = () => {};

let _nodeCounter = 1;
function makeNodeId() {
  return `node_${Date.now()}_${_nodeCounter++}`;
}

/** Regex to extract completed ref('name') calls from a Jinja2 template. */
const REF_RE = /ref\(['"]([^'"]+)['"]\)/g;

function extractRefs(source: string): string[] {
  return [...new Set([...source.matchAll(REF_RE)].map((m) => m[1]))];
}

/**
 * Build React Flow edges from the per-node ref cache.
 * Refs are the single source of truth; edges are derived, never stored separately.
 */
function computeEdges(nodes: Node[], nodeRefs: Record<string, string[]>): Edge[] {
  const nameToId = new Map(nodes.map((n) => [(n.data as PromptNodeData).label, n.id]));
  const edges: Edge[] = [];
  for (const node of nodes) {
    for (const refName of nodeRefs[node.id] ?? []) {
      const sourceId = nameToId.get(refName);
      if (sourceId && sourceId !== node.id) {
        edges.push({
          id: `${sourceId}→${node.id}`,
          source: sourceId,
          target: node.id,
          animated: true,
          markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: '#94a3b8' },
          style: { stroke: '#94a3b8', strokeWidth: 2 },
        });
      }
    }
  }
  return edges;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function DAGEditor() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>(
    (_initialSavedState?.nodes as Node[] | undefined)?.map((n) => ({
      ...n,
      data: { ...(n.data as PromptNodeData), hasOutput: false, isRunning: false },
    })) ?? [],
  );
  const [nodePrompts, setNodePrompts] = useState<Record<string, string>>(
    (_initialSavedState?.nodePrompts as Record<string, string> | undefined) ?? {},
  );
  // Per-node extracted ref list — updated whenever a prompt changes (avoids
  // running the regex over every node on every keystroke).
  const [nodeRefs, setNodeRefs] = useState<Record<string, string[]>>(
    (_initialSavedState?.nodeRefs as Record<string, string[]> | undefined) ?? {},
  );

  const [dagId, setDagId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [nodeOutputs, setNodeOutputs] = useState<Record<string, string>>({});
  const [runErrors, setRunErrors] = useState<string[]>([]);

  // Add-node dialog
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [addNodeName, setAddNodeName] = useState('');

  // Manager dialogs
  const [showDataManager, setShowDataManager] = useState(false);
  const [showFileManager, setShowFileManager] = useState(false);
  const [promptDataRows, setPromptDataRows] = useState<PromptDataRow[]>(
    (_initialSavedState?.promptDataRows as PromptDataRow[] | undefined) ?? [],
  );
  const [promptFileRows, setPromptFileRows] = useState<PromptFileRow[]>(
    // Files can't be serialised — restore id+name only, file stays null
    ((_initialSavedState?.promptFileRows as Array<{ id: string; name: string }> | undefined) ?? [])
      .map((r) => ({ ...r, file: null })),
  );

  const rfInstance = useRef<ReactFlowInstance | null>(null);

  // ── Persist to localStorage ───────────────────────────────────────────────

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          nodes: nodes.map((n) => ({
            ...n,
            data: { label: (n.data as PromptNodeData).label },
          })),
          nodePrompts,
          nodeRefs,
          promptDataRows,
          promptFileRows: promptFileRows.map(({ id, name }) => ({ id, name })),
        }),
      );
    } catch {}
  }, [nodes, nodePrompts, nodeRefs, promptDataRows, promptFileRows]);

  // ── Computed ──────────────────────────────────────────────────────────────

  const edges = useMemo(() => computeEdges(nodes, nodeRefs), [nodes, nodeRefs]);

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) ?? null,
    [nodes, selectedNodeId],
  );

  const allModelNames = useMemo(
    () => nodes.map((n) => (n.data as PromptNodeData).label),
    [nodes],
  );

  // O(1) duplicate-name check used in confirmAddNode + handleRename
  const modelNameSet = useMemo(() => new Set(allModelNames), [allModelNames]);

  const otherNodeNames = useMemo(
    () =>
      allModelNames.filter(
        (n) => n !== (selectedNode?.data as PromptNodeData | undefined)?.label,
      ),
    [allModelNames, selectedNode],
  );

  // Promptdata / promptfiles derived for the run API (only filled rows)
  const promptDataForApi = useMemo(() => {
    const entries = promptDataRows.filter((r) => r.name.trim());
    return entries.length > 0
      ? Object.fromEntries(entries.map((r) => [r.name.trim(), r.value]))
      : undefined;
  }, [promptDataRows]);

  const promptFilesForApi = useMemo(() => {
    const entries = promptFileRows.filter((r) => r.name.trim() && r.file);
    return entries.length > 0
      ? Object.fromEntries(entries.map((r) => [r.name.trim(), r.file!]))
      : undefined;
  }, [promptFileRows]);

  // ── Helpers ───────────────────────────────────────────────────────────────

  const markDirty = useCallback(() => {
    setDagId(null);
    setNodeOutputs({});
    setRunErrors([]);
  }, []);

  const updateNodeData = useCallback(
    (nodeId: string, patch: Partial<PromptNodeData>) =>
      setNodes((nds) =>
        nds.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, ...patch } } : n)),
      ),
    [setNodes],
  );

  // ── Node change handler ───────────────────────────────────────────────────

  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const removedIds = new Set(
        changes.filter((c) => c.type === 'remove').map((c) => (c as { id: string }).id),
      );
      if (removedIds.size > 0) {
        setNodePrompts((prev) => {
          const next = { ...prev };
          removedIds.forEach((id) => delete next[id]);
          return next;
        });
        setNodeRefs((prev) => {
          const next = { ...prev };
          removedIds.forEach((id) => delete next[id]);
          return next;
        });
        setSelectedNodeId((prev) => (prev && removedIds.has(prev) ? null : prev));
        markDirty();
      }
      onNodesChange(changes);
    },
    [onNodesChange, markDirty],
  );

  // ── Connect — injects ref() text into the target prompt ───────────────────

  const handleConnect = useCallback(
    (connection: Connection) => {
      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetId = connection.target;
      if (!sourceNode || !targetId) return;

      const refText = `{{ ref('${(sourceNode.data as PromptNodeData).label}') }}`;
      setNodePrompts((prev) => {
        const existing = prev[targetId] ?? '';
        if (existing.includes(refText)) return prev;
        const updated = existing ? `${existing}\n${refText}` : refText;
        setNodeRefs((r) => ({ ...r, [targetId]: extractRefs(updated) }));
        return { ...prev, [targetId]: updated };
      });
      markDirty();
    },
    [nodes, markDirty],
  );

  // ── Add node ──────────────────────────────────────────────────────────────

  const openAddDialog = useCallback(() => {
    setAddNodeName('');
    setShowAddDialog(true);
  }, []);

  const confirmAddNode = useCallback(() => {
    const name = addNodeName.trim();
    if (!name) return;
    if (/\s/.test(name)) {
      alert('Model name cannot contain spaces.');
      return;
    }
    if (modelNameSet.has(name)) {
      alert(`A model named "${name}" already exists.`);
      return;
    }
    const id = makeNodeId();
    const rawPos = { x: 200 + Math.random() * 300, y: 150 + Math.random() * 200 };
    const position = rfInstance.current
      ? rfInstance.current.screenToFlowPosition(rawPos)
      : rawPos;

    setNodes((nds) => [
      ...nds,
      {
        id,
        type: 'promptNode',
        position,
        data: { label: name, hasOutput: false, isRunning: false } satisfies PromptNodeData,
      },
    ]);
    setNodePrompts((prev) => ({ ...prev, [id]: '' }));
    setNodeRefs((prev) => ({ ...prev, [id]: [] }));
    markDirty();
    setShowAddDialog(false);
  }, [addNodeName, modelNameSet, setNodes, markDirty]);

  // ── Node selection (click and double-click share one handler) ─────────────

  const handleNodeSelect = useCallback(
    (_: React.MouseEvent, node: Node) => setSelectedNodeId(node.id),
    [],
  );

  const handlePaneClick = useCallback(() => setSelectedNodeId(null), []);

  // ── Prompt editing ────────────────────────────────────────────────────────

  const handlePromptChange = useCallback(
    (nodeId: string, value: string) => {
      setNodePrompts((prev) => ({ ...prev, [nodeId]: value }));
      setNodeRefs((prev) => ({ ...prev, [nodeId]: extractRefs(value) }));
      markDirty();
    },
    [markDirty],
  );

  const handleRename = useCallback(
    (nodeId: string, newName: string) => {
      if (/\s/.test(newName)) {
        alert('Model name cannot contain spaces.');
        return;
      }
      if (modelNameSet.has(newName)) {
        alert(`A model named "${newName}" already exists.`);
        return;
      }
      updateNodeData(nodeId, { label: newName });
      markDirty();
    },
    [modelNameSet, updateNodeData, markDirty],
  );

  // ── DAG submit ────────────────────────────────────────────────────────────

  const submitMutation = useMutation({
    mutationFn: () =>
      submitDag(
        nodes.map((n) => ({
          name: (n.data as PromptNodeData).label,
          source: nodePrompts[n.id] ?? '',
        })),
      ),
    onSuccess: (data) => {
      setDagId(data.dag_id);
      setNodeOutputs({});
      setRunErrors([]);
    },
  });

  // ── Model run ─────────────────────────────────────────────────────────────

  const runMutation = useMutation({
    mutationFn: ({ modelName, id }: { modelName: string; id: string }) =>
      runDag(id, [modelName], promptDataForApi, promptFilesForApi),

    onMutate: ({ modelName }) => {
      const nodeId = nodes.find((n) => (n.data as PromptNodeData).label === modelName)?.id;
      if (nodeId) updateNodeData(nodeId, { isRunning: true });
    },

    onSuccess: (data) => {
      setNodeOutputs((prev) => ({ ...prev, ...data.outputs }));
      if (data.errors.length > 0) setRunErrors(data.errors);

      const updatedLabels = new Set(Object.keys(data.outputs));
      setNodes((nds) =>
        nds.map((n) =>
          updatedLabels.has((n.data as PromptNodeData).label)
            ? { ...n, data: { ...n.data, isRunning: false, hasOutput: true } }
            : n,
        ),
      );
    },

    onError: (err, { modelName }) => {
      setRunErrors([(err as Error).message]);
      const nodeId = nodes.find((n) => (n.data as PromptNodeData).label === modelName)?.id;
      if (nodeId) updateNodeData(nodeId, { isRunning: false });
    },
  });

  const handleRunModel = useCallback(async () => {
    if (!selectedNode) return;
    let currentDagId = dagId;
    if (!currentDagId) {
      try {
        const result = await submitMutation.mutateAsync();
        currentDagId = result.dag_id;
      } catch {
        return; // submit error already shown in UI
      }
    }
    runMutation.mutate({
      modelName: (selectedNode.data as PromptNodeData).label,
      id: currentDagId,
    });
  }, [selectedNode, dagId, submitMutation, runMutation]);

  // ── Derived panel props ───────────────────────────────────────────────────

  const selectedModelName = selectedNode
    ? (selectedNode.data as PromptNodeData).label
    : null;

  // Derived from mutation state — eliminates separate runningModel state variable
  const isSelectedRunning =
    runMutation.isPending && runMutation.variables?.modelName === selectedModelName;

  // ── Status bar ────────────────────────────────────────────────────────────

  const statusLabel = dagId
    ? `DAG registered — id: ${dagId.slice(0, 12)}…`
    : nodes.length === 0
    ? 'Add nodes to get started'
    : 'DAG not submitted — press Submit to register';

  const promptDataCount = promptDataRows.filter((r) => r.name.trim()).length;
  const promptFileCount = promptFileRows.filter((r) => r.name.trim() && r.file).length;

  return (
    <div className="flex flex-col h-full">
      {/* ── Toolbar ── */}
      <header className="flex items-center gap-2 px-4 py-2 bg-white border-b border-border shadow-sm">
        <span className="font-bold text-foreground tracking-tight mr-2 text-sm">
          PBT DAG Editor
        </span>

        {/* Status indicator */}
        <div className="flex items-center gap-1.5 text-xs flex-1 min-w-0">
          {dagId ? (
            <CheckCircleIcon size={13} className="text-green-500 shrink-0" />
          ) : (
            <AlertCircleIcon size={13} className="text-amber-500 shrink-0" />
          )}
          <span className={`truncate ${dagId ? 'text-green-700' : 'text-amber-700'}`}>
            {statusLabel}
          </span>
        </div>

        {/* Right-side manager + action buttons */}
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowDataManager(true)}
          title="Manage promptdata() variables"
        >
          <DatabaseIcon size={13} />
          Prompt Data
          {promptDataCount > 0 && (
            <span className="ml-1 bg-primary text-primary-foreground rounded-full text-[10px] px-1.5 leading-none">
              {promptDataCount}
            </span>
          )}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowFileManager(true)}
          title="Manage promptfiles uploads"
        >
          <FileIcon size={13} />
          Prompt Files
          {promptFileCount > 0 && (
            <span className="ml-1 bg-primary text-primary-foreground rounded-full text-[10px] px-1.5 leading-none">
              {promptFileCount}
            </span>
          )}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={() => submitMutation.mutate()}
          disabled={nodes.length === 0 || submitMutation.isPending}
        >
          <SendIcon size={13} />
          {submitMutation.isPending ? 'Submitting…' : dagId ? 'Resubmit DAG' : 'Submit DAG'}
        </Button>

        <Button size="sm" onClick={openAddDialog}>
          <PlusIcon size={13} />
          Add node
        </Button>
      </header>

      {/* ── Submit error ── */}
      {submitMutation.isError && (
        <Alert variant="destructive" className="rounded-none border-x-0 border-t-0">
          <AlertDescription>{(submitMutation.error as Error).message}</AlertDescription>
        </Alert>
      )}

      {/* ── Main content ── */}
      <div className="flex flex-1 min-h-0">
        {/* React Flow canvas */}
        <div className="flex-1 min-w-0">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={handleNodesChange}
            onConnect={handleConnect}
            onConnectEnd={handleConnectEnd}
            onNodeClick={handleNodeSelect}
            onNodeDoubleClick={handleNodeSelect}
            onPaneClick={handlePaneClick}
            onInit={(instance) => { rfInstance.current = instance; }}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            deleteKeyCode="Delete"
            minZoom={0.2}
            maxZoom={3}
          >
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#e2e8f0" />
            <Controls />
            <MiniMap
              nodeColor={() => '#e2e8f0'}
              maskColor="rgba(248,250,252,0.7)"
              className="border border-border rounded-lg"
            />
          </ReactFlow>
        </div>

        {/* Node panel — key resets all local state when selected node changes,
            eliminating the need for useEffect-based draftName sync */}
        {selectedNode && selectedModelName && (
          <NodePanel
            key={selectedNode.id}
            nodeName={selectedModelName}
            prompt={nodePrompts[selectedNode.id] ?? ''}
            output={nodeOutputs[selectedModelName]}
            errors={runErrors}
            isRunning={isSelectedRunning}
            dagId={dagId}
            otherNodeNames={otherNodeNames}
            onPromptChange={(value) => handlePromptChange(selectedNode.id, value)}
            onRename={(newName) => handleRename(selectedNode.id, newName)}
            onClose={() => setSelectedNodeId(null)}
            onRun={handleRunModel}
          />
        )}
      </div>

      {/* ── Add node dialog ── */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Add model node</DialogTitle>
          </DialogHeader>
          <div>
            <label className="block text-sm text-muted-foreground mb-1.5">Model name</label>
            <Input
              autoFocus
              value={addNodeName}
              onChange={(e) => setAddNodeName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') confirmAddNode(); }}
              placeholder="e.g. article, summary, tweet"
              className="font-mono"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Lowercase letters, digits, and underscores only.
            </p>
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setShowAddDialog(false)}>Cancel</Button>
            <Button onClick={confirmAddNode} disabled={!addNodeName.trim()}>Add</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Manager dialogs ── */}
      <PromptDataManager
        open={showDataManager}
        onOpenChange={setShowDataManager}
        rows={promptDataRows}
        onRowsChange={setPromptDataRows}
      />
      <PromptFileManager
        open={showFileManager}
        onOpenChange={setShowFileManager}
        rows={promptFileRows}
        onRowsChange={setPromptFileRows}
      />
    </div>
  );
}
