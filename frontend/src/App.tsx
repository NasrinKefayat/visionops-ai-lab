import { useState } from "react";
import { useRemote } from "./api";
import { bitrate, KPI, number, State } from "./components";
import { OptimizationView } from "./OptimizationView";
import { QualityView } from "./QualityView";
import type { Monitoring } from "./types";

export default function App() {
  const [view, setView] = useState<"quality" | "optimization">("quality");
  const [refresh, setRefresh] = useState(0);
  const monitor = useRemote<Monitoring>("/api/v1/monitoring", refresh);
  const data = monitor.data;
  const connected =
    data?.database_status === "ready" && data?.publication_status === "ready";
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#main">
          <svg viewBox="0 0 36 36" aria-hidden="true">
            <path d="M4 9h8l6 17L24 9h8L21 32h-6Z" fill="currentColor" />
            <circle cx="18" cy="5" r="3" fill="#6ee7cc" />
          </svg>
          <span>
            VisionOps
            <span className="brand-subtitle">AI LAB / RESEARCH CONSOLE</span>
          </span>
        </a>
        <p className="nav-label">WORKSPACE</p>
        <nav aria-label="Dashboard sections">
          <button
            className={view === "quality" ? "active" : ""}
            aria-pressed={view === "quality"}
            onClick={() => setView("quality")}
          >
            <span aria-hidden="true">▥</span> Quality Analysis
          </button>
          <button
            className={view === "optimization" ? "active" : ""}
            aria-pressed={view === "optimization"}
            onClick={() => setView("optimization")}
          >
            <span aria-hidden="true">⌘</span> Optimization
          </button>
        </nav>
        <div className="sidebar-note">
          <span className="overline">EVIDENCE FIRST</span>
          <p>
            Measured quality.
            <br />
            Transparent predictions.
            <br />
            Reproducible research.
          </p>
          <span className="sidebar-tag">CPU-friendly · Local data</span>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <span>
            VisionOps AI Lab <span className="breadcrumb">/ Analytics</span>
          </span>
          <div>
            <span className={`connection ${connected ? "ready" : ""}`}>
              <i />
              {monitor.loading
                ? "Connecting…"
                : connected
                  ? "Data connected"
                  : "Awaiting data"}
            </span>
            <button
              onClick={() => setRefresh((value) => value + 1)}
              aria-label="Refresh dashboard data"
            >
              ↻ <span>Refresh</span>
            </button>
          </div>
        </header>
        <main id="main">
          <div className="workspace-heading">
            <div>
              <p className="eyebrow">ANALYTICS WORKSPACE</p>
              <h1>
                Encoding intelligence.
                <br />
                <span>Grounded in evidence.</span>
              </h1>
            </div>
            <p>
              Explore quality, compression and content-aware decisions across
              your published experiments.
            </p>
          </div>
          {monitor.error && (
            <State title="Monitoring unavailable" error>
              {monitor.error} Use Refresh to retry.
            </State>
          )}
          <section className="kpi-grid" aria-label="Experiment KPIs">
            <KPI
              label="Source videos"
              value={number(data?.source_videos, 0)}
              note="Published source identities"
              accent
            />
            <KPI
              label="Completed encodes"
              value={number(data?.completed_encodes, 0)}
              note="Completed published experiments"
            />
            <KPI
              label="Average measured SSIM"
              value={number(data?.average_measured_ssim, 5)}
              note="Completed quality records"
            />
            <KPI
              label="Average measured PSNR"
              value={
                data?.average_measured_psnr_db == null
                  ? "—"
                  : `${number(data.average_measured_psnr_db)} dB`
              }
              note={
                data?.infinite_psnr_count
                  ? `${data.infinite_psnr_count} infinite values excluded`
                  : "Finite measured values"
              }
            />
            <KPI
              label="Average output bitrate"
              value={bitrate(data?.average_output_bitrate_bps)}
              note="Container bitrate, including overhead"
            />
            <KPI
              label="Average encoding time"
              value={
                data?.average_encoding_seconds == null
                  ? "—"
                  : `${number(data.average_encoding_seconds, 3)} s`
              }
              note="Encoding subprocess wall time"
            />
            <KPI
              label="ML experiments"
              value={data ? number(data.available_ml_experiments, 0) : "—"}
              note="Published saved study reports"
            />
          </section>
          {data && (
            <div className="monitor-strip">
              <span>
                <i className="status-dot" />
                Experiment status
              </span>
              <span>
                {connected ? number(data.status_counts.completed ?? 0, 0) : "—"}{" "}
                completed
              </span>
              <span>
                {connected ? number(data.status_counts.running ?? 0, 0) : "—"}{" "}
                running
              </span>
              <span>
                {connected ? number(data.status_counts.failed ?? 0, 0) : "—"}{" "}
                failed
              </span>
              {data.message && <span className="muted">{data.message}</span>}
            </div>
          )}
          {view === "quality" ? (
            <QualityView refresh={refresh} />
          ) : (
            <OptimizationView refresh={refresh} />
          )}
          <footer>
            VisionOps AI Lab{" "}
            <span>
              Research tools, not clinical validation. Missing values are shown
              as —.
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
