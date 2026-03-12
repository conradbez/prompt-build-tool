import {
  useCallback,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  addEdge,
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
import { PlusIcon, SendIcon, AlertCircleIcon, CheckCircleIcon } from 'lucide-react';

import PromptNode, { type PromptNodeData } from './PromptNode';
import NodePanel from './NodePanel';
import { submitDag, runDag } from '../api';

const nodeTypes = { promptNode: PromptNode };

let nodeCounter = 1;

function makeNodeId() {
  return `node_${Date.now()}_${nodeCounter++}`;
}

/** Extract ref('name') calls from a Jinja2 prompt source. */
function extractRefs(source: string): string[] {
  const matches = [...source.matchAll(/ref\(['"]([^'"]+)['"]\)/g)];
  return [...new Set(matches.map((m) => m[1]))];
}

/** Build React Flow edges from nodePrompts using ref() calls as source of truth. */
function computeEdges(
  nodes: Node[],
  nodePrompts: Record<string, string>,
): Edge[] {
  const nameToId = new Map(nodes.map((n) => [(n.data as PromptNodeData).label, n.id]));
  const edges: Edge[] = [];

  for (const node of nodes) {
    const source = nodePrompts[node.id] ?? '';
    const refs = extractRefs(source);
    for (const refName of refs) {
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

export default function DAGEditor() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [nodePrompts, setNodePrompts] = useState<Record<string, string>>({});
  const [dagId, setDagId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [nodeOutputs, setNodeOutputs] = useState<Record<string, string>>({});
  const [runningModel, setRunningModel] = useState<string | null>(null);
  const [runErrors, setRunErrors] = useState<string[]>([]);
  const [addNodeName, setAddNodeName] = useState('');
  const [showAddDialog, setShowAddDialog] = useState(false);
  const addInputRef = useRef<HTMLInputElement>(null);
  const rfInstance = useRef<ReactFlowInstance | null>(null);

  // ── Computed values ──────────────────────────────────────────────────────

  const edges = useMemo(() => computeEdges(nodes, nodePrompts), [nodes, nodePrompts]);

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) ?? null,
    [nodes, selectedNodeId],
  );

  const allModelNames = useMemo(
    () => nodes.map((n) => (n.data as PromptNodeData).label),
    [nodes],
  );

  // ── Helpers ──────────────────────────────────────────────────────────────

  /** Mark the DAG as dirty (requires re-submit) on structural changes. */
  const markDirty = useCallback(() => {
    setDagId(null);
    setNodeOutputs({});
    setRunErrors([]);
  }, []);

  const updateNodeData = useCallback(
    (nodeId: string, data: Partial<PromptNodeData>) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n,
        ),
      );
    },
    [setNodes],
  );

  // ── Node / Edge change handlers ──────────────────────────────────────────

  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const hasRemoval = changes.some((c) => c.type === 'remove');
      if (hasRemoval) {
        const removedIds = new Set(
          changes.filter((c) => c.type === 'remove').map((c) => (c as { id: string }).id),
        );
        setNodePrompts((prev) => {
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

  /** When the user drags a handle connection, inject ref() into the target prompt. */
  const handleConnect = useCallback(
    (connection: Connection) => {
      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetId = connection.target;
      if (!sourceNode || !targetId) return;

      const refText = `{{ ref('${(sourceNode.data as PromptNodeData).label}') }}`;
      setNodePrompts((prev) => {
        const existing = prev[targetId] ?? '';
        // Avoid duplicate refs
        if (existing.includes(refText)) return prev;
        return { ...prev, [targetId]: existing ? `${existing}\n${refText}` : refText };
      });
      markDirty();
      // addEdge is called here only to trigger the connection visually;
      // actual edges are still derived from prompts.
      void addEdge(connection, []);
    },
    [nodes, markDirty],
  );

  // Suppress the default "drop on pane to create node" cursor change
  const handleConnectEnd: OnConnectEnd = useCallback(() => {}, []);

  // ── Add node ─────────────────────────────────────────────────────────────

  const openAddDialog = () => {
    setAddNodeName('');
    setShowAddDialog(true);
    setTimeout(() => addInputRef.current?.focus(), 50);
  };

  const confirmAddNode = useCallback(() => {
    const name = addNodeName.trim();
    if (!name) return;
    if (allModelNames.includes(name)) {
      alert(`A model named "${name}" already exists.`);
      return;
    }
    const id = makeNodeId();
    const position = rfInstance.current
      ? rfInstance.current.screenToFlowPosition({ x: 200 + Math.random() * 300, y: 150 + Math.random() * 200 })
      : { x: 200 + Math.random() * 300, y: 150 + Math.random() * 200 };

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
    markDirty();
    setShowAddDialog(false);
  }, [addNodeName, allModelNames, setNodes, markDirty]);

  // ── Node selection / editing ─────────────────────────────────────────────

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
    },
    [],
  );

  const handleNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
    },
    [],
  );

  const handlePaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  // ── Prompt editing ───────────────────────────────────────────────────────

  const handlePromptChange = useCallback(
    (nodeId: string, value: string) => {
      setNodePrompts((prev) => ({ ...prev, [nodeId]: value }));
      markDirty();
    },
    [markDirty],
  );

  const handleRename = useCallback(
    (nodeId: string, newName: string) => {
      if (allModelNames.includes(newName)) {
        alert(`A model named "${newName}" already exists.`);
        return;
      }
      updateNodeData(nodeId, { label: newName });
      markDirty();
    },
    [allModelNames, updateNodeData, markDirty],
  );

  // ── DAG submit ───────────────────────────────────────────────────────────

  const submitMutation = useMutation({
    mutationFn: () => {
      const payload = nodes.map((n) => ({
        name: (n.data as PromptNodeData).label,
        source: nodePrompts[n.id] ?? '',
      }));
      return submitDag(payload);
    },
    onSuccess: (data) => {
      setDagId(data.dag_id);
      setNodeOutputs({});
      setRunErrors([]);
    },
  });

  // ── Model run ────────────────────────────────────────────────────────────

  const runMutation = useMutation({
    mutationFn: (modelName: string) => runDag(dagId!, [modelName]),
    onMutate: (modelName) => {
      setRunningModel(modelName);
      setRunErrors([]);
      // Update node visual state
      const nodeId = nodes.find(
        (n) => (n.data as PromptNodeData).label === modelName,
      )?.id;
      if (nodeId) updateNodeData(nodeId, { isRunning: true });
    },
    onSuccess: (data, modelName) => {
      setNodeOutputs((prev) => ({ ...prev, ...data.outputs }));
      if (data.errors.length > 0) setRunErrors(data.errors);

      // Update hasOutput / isRunning flags for all affected nodes
      for (const [name] of Object.entries(data.outputs)) {
        const nodeId = nodes.find((n) => (n.data as PromptNodeData).label === name)?.id;
        if (nodeId) updateNodeData(nodeId, { isRunning: false, hasOutput: true });
      }
      void modelName; // consumed above
    },
    onError: (err, modelName) => {
      setRunErrors([(err as Error).message]);
      const nodeId = nodes.find(
        (n) => (n.data as PromptNodeData).label === modelName,
      )?.id;
      if (nodeId) updateNodeData(nodeId, { isRunning: false });
    },
    onSettled: () => setRunningModel(null),
  });

  const handleRunModel = useCallback(() => {
    if (!selectedNode || !dagId) return;
    runMutation.mutate((selectedNode.data as PromptNodeData).label);
  }, [selectedNode, dagId, runMutation]);

  // ── Derived panel props ──────────────────────────────────────────────────

  const selectedModelName = selectedNode
    ? (selectedNode.data as PromptNodeData).label
    : null;

  const otherNodeNames = useMemo(
    () => allModelNames.filter((n) => n !== selectedModelName),
    [allModelNames, selectedModelName],
  );

  // ── Status bar info ──────────────────────────────────────────────────────

  const statusLabel = dagId
    ? `DAG registered — id: ${dagId.slice(0, 12)}…`
    : nodes.length === 0
    ? 'Add nodes to get started'
    : 'DAG not submitted — press Submit to register';

  return (
    <div className="flex flex-col h-full">
      {/* ── Top toolbar ── */}
      <header className="flex items-center gap-3 px-4 py-2 bg-white border-b border-slate-200 shadow-sm">
        <span className="font-bold text-slate-800 tracking-tight mr-2">PBT DAG Editor</span>

        {/* Status indicator */}
        <div className="flex items-center gap-1.5 text-xs flex-1">
          {dagId ? (
            <CheckCircleIcon size={13} className="text-green-500" />
          ) : (
            <AlertCircleIcon size={13} className="text-amber-500" />
          )}
          <span className={dagId ? 'text-green-700' : 'text-amber-700'}>{statusLabel}</span>
        </div>

        {/* Submit DAG */}
        <button
          onClick={() => submitMutation.mutate()}
          disabled={nodes.length === 0 || submitMutation.isPending}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:bg-slate-300 text-white text-sm font-medium rounded-lg transition-colors"
        >
          <SendIcon size={13} />
          {submitMutation.isPending ? 'Submitting…' : dagId ? 'Resubmit DAG' : 'Submit DAG'}
        </button>

        {/* Add node */}
        <button
          onClick={openAddDialog}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
        >
          <PlusIcon size={13} />
          Add node
        </button>
      </header>

      {/* ── Submit error ── */}
      {submitMutation.isError && (
        <div className="px-4 py-2 bg-red-50 border-b border-red-200 text-xs text-red-700">
          {(submitMutation.error as Error).message}
        </div>
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
            onNodeClick={handleNodeClick}
            onNodeDoubleClick={handleNodeDoubleClick}
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
              className="border border-slate-200 rounded-lg"
            />
          </ReactFlow>
        </div>

        {/* Node panel (shown when a node is selected) */}
        {selectedNode && selectedModelName && (
          <NodePanel
            nodeId={selectedNode.id}
            nodeName={selectedModelName}
            prompt={nodePrompts[selectedNode.id] ?? ''}
            output={nodeOutputs[selectedModelName]}
            errors={runErrors}
            isRunning={runningModel === selectedModelName}
            dagId={dagId}
            otherNodeNames={otherNodeNames}
            onPromptChange={handlePromptChange}
            onRename={handleRename}
            onClose={() => setSelectedNodeId(null)}
            onRun={handleRunModel}
          />
        )}
      </div>

      {/* ── Add node dialog ── */}
      {showAddDialog && (
        <div
          className="fixed inset-0 bg-black/30 flex items-center justify-center z-50"
          onClick={() => setShowAddDialog(false)}
        >
          <div
            className="bg-white rounded-xl shadow-xl p-6 w-80"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-base font-semibold text-slate-800 mb-3">Add model node</h2>
            <label className="block text-sm text-slate-600 mb-1">Model name</label>
            <input
              ref={addInputRef}
              type="text"
              value={addNodeName}
              onChange={(e) => setAddNodeName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') confirmAddNode();
                if (e.key === 'Escape') setShowAddDialog(false);
              }}
              placeholder="e.g. article, summary, tweet"
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
            <p className="text-xs text-slate-400 mt-1">
              Use lowercase letters, digits, and underscores.
            </p>
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => setShowAddDialog(false)}
                className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={confirmAddNode}
                disabled={!addNodeName.trim()}
                className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium rounded-lg"
              >
                Add
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
