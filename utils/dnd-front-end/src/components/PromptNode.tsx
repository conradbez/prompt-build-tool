import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';

export type PromptNodeData = {
  label: string;
  hasOutput: boolean;
  isRunning: boolean;
};

function PromptNode({ data, selected }: NodeProps) {
  const d = data as PromptNodeData;

  return (
    <div
      className={[
        'relative flex flex-col items-center rounded-xl px-5 py-3 shadow-md cursor-pointer',
        'bg-white border-2 transition-all duration-150 min-w-[140px]',
        selected
          ? 'border-blue-500 shadow-blue-100 shadow-lg'
          : 'border-slate-200 hover:border-slate-400',
      ].join(' ')}
    >
      {/* target handle (top – receives connections from upstream models) */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !border-2 !border-slate-400 !bg-white"
      />

      {/* Node label */}
      <div className="font-mono font-semibold text-sm text-slate-800 truncate max-w-[200px]">
        {d.label}
      </div>

      {/* Status badge */}
      <div className="mt-1 flex items-center gap-1">
        {d.isRunning ? (
          <span className="inline-flex items-center gap-1 text-[10px] text-amber-600 font-medium">
            <span className="animate-spin">⟳</span> running
          </span>
        ) : d.hasOutput ? (
          <span className="inline-flex items-center gap-1 text-[10px] text-green-600 font-medium">
            ✓ done
          </span>
        ) : (
          <span className="text-[10px] text-slate-400">double-click to edit</span>
        )}
      </div>

      {/* source handle (bottom – this model feeds into downstream models) */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !border-2 !border-slate-400 !bg-white"
      />
    </div>
  );
}

export default memo(PromptNode);
