import { ReactFlowProvider } from '@xyflow/react';
import DAGEditor from './components/DAGEditor';

export default function App() {
  return (
    <div className="h-screen flex flex-col bg-slate-50">
      <ReactFlowProvider>
        <DAGEditor />
      </ReactFlowProvider>
    </div>
  );
}
