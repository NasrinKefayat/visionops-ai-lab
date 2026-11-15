import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

function App() {
  return (
    <main>
      <p className="eyebrow">VisionOps-AI · Phase 1</p>
      <h1>Video & medical image intelligence</h1>
      <p>A modular foundation for reproducible AI experiments and analysis.</p>
      <section aria-labelledby="workspace-title">
        <h2 id="workspace-title">Analysis workspace</h2>
        <p>Video processing, medical imaging, experiment charts, and model KPIs are planned for future phases.</p>
        <p>No datasets or trained models are connected.</p>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
