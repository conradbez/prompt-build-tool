import { useRef, useState, useCallback, useEffect } from 'react';
import { PlayIcon, RefreshCwIcon, XIcon } from 'lucide-react';

interface NodePanelProps {
  nodeId: string;
  nodeName: string;
  prompt: string;
  output: string | undefined;
  errors: string[];
  isRunning: boolean;
  dagId: string | null;
  otherNodeNames: string[];
  onPromptChange: (nodeId: string, value: string) => void;
  onRename: (nodeId: string, newName: string) => void;
  onClose: () => void;
  onRun: () => void;
}

/** Detect if the cursor sits inside a ref(' ... ') expression and return the partial name. */
function getRefPartial(text: string, cursorPos: number): string | null {
  const before = text.slice(0, cursorPos);
  const match = before.match(/ref\(['"]?([^'")\s]*)$/);
  return match ? match[1] : null;
}

export default function NodePanel({
  nodeId,
  nodeName,
  prompt,
  output,
  errors,
  isRunning,
  dagId,
  otherNodeNames,
  onPromptChange,
  onRename,
  onClose,
  onRun,
}: NodePanelProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selectedSuggestion, setSelectedSuggestion] = useState(0);
  const [isEditingName, setIsEditingName] = useState(false);
  const [draftName, setDraftName] = useState(nodeName);

  // Reset draft name when nodeName changes (e.g. switching nodes)
  useEffect(() => {
    setDraftName(nodeName);
    setIsEditingName(false);
  }, [nodeName]);

  const handleTextChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const value = e.target.value;
      onPromptChange(nodeId, value);

      const partial = getRefPartial(value, e.target.selectionStart ?? value.length);
      if (partial !== null) {
        const filtered = otherNodeNames.filter((n) =>
          n.toLowerCase().startsWith(partial.toLowerCase()),
        );
        setSuggestions(filtered);
        setSelectedSuggestion(0);
      } else {
        setSuggestions([]);
      }
    },
    [nodeId, onPromptChange, otherNodeNames],
  );

  const insertSuggestion = useCallback(
    (name: string) => {
      const textarea = textareaRef.current;
      if (!textarea) return;
      const cursorPos = textarea.selectionStart ?? prompt.length;
      const before = prompt.slice(0, cursorPos);
      const after = prompt.slice(cursorPos);
      // Replace the partial ref pattern with the completed ref() call
      const newBefore = before.replace(/ref\(['"]?[^'")\s]*$/, `ref('${name}')`);
      const newValue = newBefore + after;
      onPromptChange(nodeId, newValue);
      setSuggestions([]);
      // Restore focus and set cursor after the inserted text
      setTimeout(() => {
        textarea.focus();
        textarea.selectionStart = textarea.selectionEnd = newBefore.length;
      }, 0);
    },
    [nodeId, prompt, onPromptChange],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (suggestions.length === 0) return;
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedSuggestion((i) => Math.min(i + 1, suggestions.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedSuggestion((i) => Math.max(i - 1, 0));
      } else if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        insertSuggestion(suggestions[selectedSuggestion]);
      } else if (e.key === 'Escape') {
        setSuggestions([]);
      }
    },
    [suggestions, selectedSuggestion, insertSuggestion],
  );

  const commitRename = () => {
    const trimmed = draftName.trim();
    if (trimmed && trimmed !== nodeName) {
      onRename(nodeId, trimmed);
    }
    setIsEditingName(false);
  };

  return (
    <div className="flex flex-col h-full bg-white border-l border-slate-200 w-[420px] flex-shrink-0">
      {/* Panel header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-slate-50">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
            Model
          </span>
          {isEditingName ? (
            <input
              autoFocus
              className="font-mono font-semibold text-sm border-b border-blue-400 outline-none bg-transparent flex-1 min-w-0"
              value={draftName}
              onChange={(e) => setDraftName(e.target.value)}
              onBlur={commitRename}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitRename();
                if (e.key === 'Escape') {
                  setDraftName(nodeName);
                  setIsEditingName(false);
                }
              }}
            />
          ) : (
            <button
              className="font-mono font-semibold text-sm text-slate-800 hover:text-blue-600 truncate"
              onClick={() => setIsEditingName(true)}
              title="Click to rename"
            >
              {nodeName}
            </button>
          )}
        </div>
        <button
          onClick={onClose}
          className="ml-2 p-1 rounded hover:bg-slate-200 text-slate-400 hover:text-slate-600 transition-colors"
          title="Close panel"
        >
          <XIcon size={14} />
        </button>
      </div>

      {/* Prompt editor */}
      <div className="px-4 pt-3 pb-2">
        <label className="block text-xs font-medium text-slate-500 mb-1">
          Prompt template
          <span className="ml-1 text-slate-400 font-normal">
            — use <code className="bg-slate-100 px-1 rounded">{'{{ ref(\'name\') }}'}</code> to
            reference other models
          </span>
        </label>

        <div className="relative">
          <textarea
            ref={textareaRef}
            value={prompt}
            onChange={handleTextChange}
            onKeyDown={handleKeyDown}
            rows={10}
            spellCheck={false}
            className="w-full font-mono text-sm border border-slate-200 rounded-lg p-3 resize-none focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent placeholder-slate-300 leading-relaxed"
            placeholder={`Write a Jinja2 prompt template.\n\nExample:\nWrite an article about {{ promptdata('topic') }}\n\nOr reference another model:\n{{ ref('article') }}`}
          />

          {/* Autocomplete dropdown */}
          {suggestions.length > 0 && (
            <div className="autocomplete-list">
              {suggestions.map((name, idx) => (
                <div
                  key={name}
                  className={`autocomplete-item ${idx === selectedSuggestion ? 'selected' : ''}`}
                  onMouseDown={(e) => {
                    e.preventDefault(); // Don't blur textarea
                    insertSuggestion(name);
                  }}
                >
                  ref('{name}')
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Autocomplete hint */}
        <p className="text-[11px] text-slate-400 mt-1">
          Type <code>ref(&#39;</code> to autocomplete a model name from the DAG.
        </p>
      </div>

      {/* Run button */}
      <div className="px-4 py-2 border-t border-slate-100">
        {dagId ? (
          <button
            onClick={onRun}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white text-sm font-medium rounded-lg transition-colors"
          >
            {isRunning ? (
              <>
                <RefreshCwIcon size={14} className="animate-spin" />
                Running…
              </>
            ) : (
              <>
                <PlayIcon size={14} />
                Run model
              </>
            )}
          </button>
        ) : (
          <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
            Submit the DAG first to enable running individual models.
          </p>
        )}
      </div>

      {/* Errors */}
      {errors.length > 0 && (
        <div className="mx-4 my-2 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-xs font-semibold text-red-700 mb-1">Errors</p>
          {errors.map((e, i) => (
            <p key={i} className="text-xs text-red-600 font-mono">
              {e}
            </p>
          ))}
        </div>
      )}

      {/* Output */}
      <div className="flex-1 overflow-hidden flex flex-col px-4 pb-4 min-h-0">
        <div className="border-t border-slate-100 pt-3 flex-1 flex flex-col min-h-0">
          <label className="block text-xs font-medium text-slate-500 mb-2">
            Output
            {output && (
              <span className="ml-2 text-green-600 font-normal">✓ ready</span>
            )}
          </label>
          {output ? (
            <pre className="flex-1 overflow-auto font-mono text-xs bg-slate-50 border border-slate-200 rounded-lg p-3 whitespace-pre-wrap text-slate-700 leading-relaxed">
              {output}
            </pre>
          ) : (
            <div className="flex-1 flex items-center justify-center bg-slate-50 border border-dashed border-slate-200 rounded-lg">
              <p className="text-sm text-slate-400">
                {isRunning ? 'Running model…' : 'No output yet — run the model to see results here.'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
