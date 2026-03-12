import { useRef, useState, useCallback } from 'react';
import { PlayIcon, RefreshCwIcon, XIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Card, CardContent } from '@/components/ui/card';

// Props — nodeId removed; callbacks are pre-bound in DAGEditor so the panel
// doesn't need to know its own identity.
interface NodePanelProps {
  nodeName: string;
  prompt: string;
  output: string | undefined;
  errors: string[];
  isRunning: boolean;
  otherNodeNames: string[];
  promptDataNames: string[];
  promptFileNames: string[];
  onPromptChange: (value: string) => void;
  onRename: (newName: string) => void;
  onClose: () => void;
  onRun: () => void;
}

interface AcSuggestion {
  display: string;
  insert: string;
  replaceFrom: number;
}

/**
 * Return autocomplete suggestions for ref(), promptdata(), and promptfiles() calls.
 * Triggers on:
 *   - partial function name at a word boundary (e.g. "re" → ref(...))
 *   - open paren with no quote (e.g. "ref(par")
 *   - inside quotes with a partial name (e.g. "ref('par", "promptdata(\"key")
 * Quote style is normalised to single quotes on insertion.
 */
function getAcSuggestions(
  text: string,
  cursor: number,
  nodeNames: string[],
  promptDataNames: string[],
  promptFileNames: string[],
): AcSuggestion[] {
  const before = text.slice(0, cursor);

  const make = (func: string, name: string, from: number): AcSuggestion => ({
    display: `${func}('${name}')`,
    insert: `${func}('${name}')`,
    replaceFrom: from,
  });

  // Case 1: inside quotes — ref('partial | promptdata("partial
  const m1 = before.match(/(ref|promptdata|promptfiles)\(['"]([^'")\s]*)$/);
  if (m1) {
    const [full, func, partial] = m1;
    const from = cursor - full.length;
    const pool = func === 'ref' ? nodeNames : func === 'promptdata' ? promptDataNames : promptFileNames;
    return pool
      .filter(n => n.toLowerCase().startsWith(partial.toLowerCase()))
      .map(n => make(func, n, from));
  }

  // Case 2: open paren, no quote yet — ref(partial | promptdata(
  const m2 = before.match(/(ref|promptdata|promptfiles)\(([^'")\s]*)$/);
  if (m2) {
    const [full, func, partial] = m2;
    const from = cursor - full.length;
    const pool = func === 'ref' ? nodeNames : func === 'promptdata' ? promptDataNames : promptFileNames;
    return pool
      .filter(n => !partial || n.toLowerCase().startsWith(partial.toLowerCase()))
      .map(n => make(func, n, from));
  }

  // Case 3: partial function name at word boundary — "re", "promptd", etc.
  const m3 = before.match(/(?:^|[\s{%(\[,\n])([a-z][a-z]*)$/);
  if (!m3) return [];
  const typed = m3[1];
  const from = cursor - typed.length;
  const results: AcSuggestion[] = [];
  if ('ref'.startsWith(typed))        nodeNames.forEach(n => results.push(make('ref', n, from)));
  if ('promptdata'.startsWith(typed))  promptDataNames.forEach(n => results.push(make('promptdata', n, from)));
  if ('promptfiles'.startsWith(typed)) promptFileNames.forEach(n => results.push(make('promptfiles', n, from)));
  return results;
}

export default function NodePanel({
  nodeName,
  prompt,
  output,
  errors,
  isRunning,
  otherNodeNames,
  promptDataNames,
  promptFileNames,
  onPromptChange,
  onRename,
  onClose,
  onRun,
}: NodePanelProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [suggestions, setSuggestions] = useState<AcSuggestion[]>([]);
  const [activeSuggestion, setActiveSuggestion] = useState(0);
  const [promptHeight, setPromptHeight] = useState(240);
  const dragStart = useRef<{ y: number; h: number } | null>(null);

  const handleDragMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragStart.current = { y: e.clientY, h: promptHeight };
    const onMove = (ev: MouseEvent) => {
      if (!dragStart.current) return;
      const delta = ev.clientY - dragStart.current.y;
      setPromptHeight(Math.max(80, dragStart.current.h + delta));
    };
    const onUp = () => {
      dragStart.current = null;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [promptHeight]);

  // draftName is initialized once per mount; the parent passes key={nodeId} so
  // this component remounts when the selected node changes — no useEffect needed.
  const [isEditingName, setIsEditingName] = useState(false);
  const [draftName, setDraftName] = useState(nodeName);

  // ── Textarea change + autocomplete detection ──────────────────────────────

  const handleTextChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const value = e.target.value;
      onPromptChange(value);
      const cursor = e.target.selectionStart ?? value.length;
      const next = getAcSuggestions(value, cursor, otherNodeNames, promptDataNames, promptFileNames);
      setSuggestions(next);
      setActiveSuggestion(0);
    },
    [onPromptChange, otherNodeNames, promptDataNames, promptFileNames],
  );

  const insertSuggestion = useCallback(
    (suggestion: AcSuggestion) => {
      const textarea = textareaRef.current;
      if (!textarea) return;
      const cursorPos = textarea.selectionStart ?? prompt.length;
      const newText = prompt.slice(0, suggestion.replaceFrom) + suggestion.insert + prompt.slice(cursorPos);
      const newCursor = suggestion.replaceFrom + suggestion.insert.length;
      onPromptChange(newText);
      setSuggestions([]);
      setTimeout(() => {
        textarea.focus();
        textarea.selectionStart = textarea.selectionEnd = newCursor;
      }, 0);
    },
    [prompt, onPromptChange],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (suggestions.length === 0) return;
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveSuggestion((i) => Math.min(i + 1, suggestions.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveSuggestion((i) => Math.max(i - 1, 0));
      } else if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        insertSuggestion(suggestions[activeSuggestion]);
      } else if (e.key === 'Escape') {
        setSuggestions([]);
      }
    },
    [suggestions, activeSuggestion, insertSuggestion],
  );

  // ── Rename ────────────────────────────────────────────────────────────────

  const commitRename = () => {
    const trimmed = draftName.trim();
    if (trimmed && trimmed !== nodeName) onRename(trimmed);
    setIsEditingName(false);
  };

  return (
    <div className="flex flex-col h-full bg-white border-l border-border w-[420px] flex-shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Model
          </span>
          {isEditingName ? (
            <Input
              autoFocus
              className="font-mono font-semibold text-sm h-7 py-0 border-0 border-b rounded-none focus-visible:ring-0 bg-transparent flex-1 min-w-0"
              value={draftName}
              onChange={(e) => setDraftName(e.target.value)}
              onBlur={commitRename}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitRename();
                if (e.key === 'Escape') { setDraftName(nodeName); setIsEditingName(false); }
              }}
            />
          ) : (
            <Button
              variant="ghost"
              size="sm"
              className="font-mono font-semibold text-sm h-auto py-0 px-1 hover:text-primary truncate"
              onClick={() => setIsEditingName(true)}
              title="Click to rename"
            >
              {nodeName}
            </Button>
          )}
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} title="Close panel">
          <XIcon size={14} />
        </Button>
      </div>

      {/* Prompt editor */}
      <div className="px-4 pt-3 pb-0">
        <label className="block text-xs font-medium text-muted-foreground mb-1">
          Prompt template
          <span className="ml-1 font-normal">
            — use{' '}
            <code className="bg-muted px-1 rounded text-[11px]">{'{{ ref(\'name\') }}'}</code>{' '}
            to reference other models
          </span>
        </label>

        <div className="relative">
          <Textarea
            ref={textareaRef}
            value={prompt}
            onChange={handleTextChange}
            onKeyDown={handleKeyDown}
            style={{ height: promptHeight }}
            spellCheck={false}
            className="font-mono text-sm resize-none leading-relaxed"
            placeholder={`Write a Jinja2 prompt template.\n\nExample:\nWrite an article about {{ promptdata('topic') }}\n\nOr reference another model:\n{{ ref('article') }}`}
          />

          {/* Autocomplete dropdown */}
          {suggestions.length > 0 && (
            <div className="autocomplete-list">
              {suggestions.map((s, idx) => (
                <div
                  key={s.display + idx}
                  className={`autocomplete-item ${idx === activeSuggestion ? 'selected' : ''}`}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    insertSuggestion(s);
                  }}
                >
                  {s.display}
                </div>
              ))}
            </div>
          )}
        </div>

        <p className="text-[11px] text-muted-foreground mt-1 mb-0">
          Type <code className="bg-muted px-0.5 rounded">ref(&#39;</code> to autocomplete a model
          name. Arrow keys to navigate, Enter/Tab to insert.
        </p>
      </div>

      {/* Drag handle */}
      <div
        onMouseDown={handleDragMouseDown}
        className="h-2 mx-4 my-1 rounded cursor-row-resize flex items-center justify-center group"
        title="Drag to resize"
      >
        <div className="w-8 h-0.5 rounded-full bg-border group-hover:bg-muted-foreground transition-colors" />
      </div>

      {/* Run button */}
      <div className="px-4 py-2 border-t border-border">
        <Button onClick={onRun} disabled={isRunning} size="sm">
          {isRunning ? (
            <><RefreshCwIcon size={13} className="animate-spin" /> Running…</>
          ) : (
            <><PlayIcon size={13} /> Run model</>
          )}
        </Button>
      </div>

      {/* Errors */}
      {errors.length > 0 && (
        <div className="px-4 py-2">
          <Alert variant="destructive">
            <AlertTitle>Run errors</AlertTitle>
            <AlertDescription>
              <div className="max-h-32 overflow-y-auto mt-1">
                {errors.map((e, i) => (
                  <p key={i} className="font-mono text-xs whitespace-pre-wrap">
                    {e}
                  </p>
                ))}
              </div>
            </AlertDescription>
          </Alert>
        </div>
      )}

      {/* Output */}
      <div className="flex-1 overflow-hidden flex flex-col px-4 pb-4 min-h-0">
        <div className="border-t border-border pt-3 flex-1 flex flex-col min-h-0">
          <label className="block text-xs font-medium text-muted-foreground mb-2">
            Output
            {output && <span className="ml-2 text-green-600 font-normal">✓ ready</span>}
          </label>

          {output ? (
            <Card className="flex-1 overflow-auto">
              <CardContent className="p-3">
                <pre className="font-mono text-xs whitespace-pre-wrap text-foreground leading-relaxed">
                  {output}
                </pre>
              </CardContent>
            </Card>
          ) : (
            <Card className="flex-1 flex items-center justify-center border-dashed">
              <p className="text-sm text-muted-foreground">
                {isRunning
                  ? 'Running model…'
                  : 'No output yet — run the model to see results here.'}
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
