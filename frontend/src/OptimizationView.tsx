import { useEffect, useRef, useState } from "react";
import { query, request, useRemote } from "./api";
import { BarChart } from "./charts";
import {
  bitrate,
  KindBadge,
  number,
  Pagination,
  shortId,
  State,
} from "./components";
import type { Page, Recommendation, Study, StudySummary, Video } from "./types";

const policies = ["fixed", "content_aware", "oracle"];
const policyName = (name: string) =>
  ({ fixed: "Fixed CRF", content_aware: "Content-aware ML", oracle: "Oracle" })[
    name
  ] || name;
const modelName = (name: string) =>
  name === "random_forest" ? "Random forest" : "Mean baseline";

export function OptimizationView({ refresh }: { refresh: number }) {
  const [offset, setOffset] = useState(0);
  const [studyId, setStudyId] = useState<string | null>(null);
  const [target, setTarget] = useState("ssim");
  const [videoOffset, setVideoOffset] = useState(0);
  const [videoId, setVideoId] = useState("");
  const [threshold, setThreshold] = useState("0.99");
  const [prediction, setPrediction] = useState<Recommendation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const pending = useRef<AbortController | null>(null);
  const studies = useRemote<Page<StudySummary>>(
    query("studies", { offset, limit: 20 }),
    refresh,
  );
  const study = useRemote<Study>(
    studyId ? `/api/v1/studies/${studyId}` : null,
    refresh,
  );
  const videos = useRemote<Page<Video>>(
    query("videos", { offset: videoOffset, limit: 20 }),
    refresh,
  );
  useEffect(() => {
    if (studies.data && !studies.data.items.some((s) => s.id === studyId))
      setStudyId(studies.data.items[0]?.id ?? null);
  }, [studies.data, studyId]);
  useEffect(() => {
    pending.current?.abort();
    setPrediction(null);
    setError(null);
    setBusy(false);
    return () => pending.current?.abort();
  }, [studyId, videoId, threshold]);
  useEffect(() => {
    if (videos.data && !videos.data.items.some((v) => String(v.id) === videoId))
      setVideoId(String(videos.data.items[0]?.id ?? ""));
  }, [videos.data, videoId]);
  async function infer(event: React.FormEvent) {
    event.preventDefault();
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true);
    setError(null);
    setPrediction(null);
    try {
      const result = await request<Recommendation>(
        `/api/v1/models/${studyId}/recommend`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            video_id: Number(videoId),
            minimum_ssim: Number(threshold),
          }),
          signal: controller.signal,
        },
      );
      if (!controller.signal.aborted) setPrediction(result);
    } catch (error) {
      if (!controller.signal.aborted)
        setError(
          error instanceof Error ? error.message : "Prediction is unavailable.",
        );
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }
  const selected = study.data;
  const importance = selected?.feature_importance[target] || {};
  const reduction =
    selected?.policies.content_aware?.mean_bitrate_reduction_percent_vs_fixed;
  return (
    <>
      <div className="section-heading">
        <div>
          <p className="eyebrow">02 / CONTENT-AWARE OPTIMIZATION</p>
          <h2>Predictions meet evidence.</h2>
          <p>
            Inspect saved models, quality constraints, and held-out policy
            comparisons.
          </p>
        </div>
        <span className="badge predicted">
          Predictions are not measurements
        </span>
      </div>
      <section className="panel filters">
        <label className="wide">
          Research experiment
          <select
            aria-label="Research experiment"
            value={studyId ?? ""}
            onChange={(e) => setStudyId(e.target.value)}
            disabled={!studies.data?.items.length}
          >
            <option value="" disabled>
              Select a research experiment
            </option>
            {studies.data?.items.map((s) => (
              <option key={s.id} value={s.id}>
                Study {shortId(s.id)} ·{" "}
                {s.dataset_kind === "synthetic_only"
                  ? "Synthetic"
                  : "Public video"}{" "}
                · {s.source_count} sources
              </option>
            ))}
          </select>
        </label>
        <Pagination
          offset={offset}
          limit={20}
          total={studies.data?.total ?? 0}
          onChange={setOffset}
          label="studies"
        />
      </section>
      {studies.loading ? (
        <State title="Loading research experiments…" />
      ) : studies.error ? (
        <State title="Research experiments unavailable" error>
          {studies.error}
        </State>
      ) : !studies.data?.items.length ? (
        <State title="No published ML experiments yet">
          Run the documented Phase 3 study and publish its public reports to
          inspect models and comparisons.
        </State>
      ) : study.loading ? (
        <State title="Loading study…" />
      ) : study.error ? (
        <State title="Study unavailable" error>
          {study.error}
        </State>
      ) : (
        selected && (
          <>
            <section className="panel source-summary">
              <div>
                <span className="overline">SAVED MODEL BUNDLE</span>
                <h3>Study {shortId(selected.id)}</h3>
                <KindBadge kind={selected.dataset_kind} />
              </div>
              <dl>
                <div>
                  <dt>SSIM model</dt>
                  <dd>{modelName(selected.selected_models.ssim)}</dd>
                </div>
                <div>
                  <dt>Bitrate model</dt>
                  <dd>{modelName(selected.selected_models.bitrate_bps)}</dd>
                </div>
                <div>
                  <dt>Train / validation / test groups</dt>
                  <dd>
                    {selected.split_groups.train} /{" "}
                    {selected.split_groups.validation} /{" "}
                    {selected.split_groups.test}
                  </dd>
                </div>
                <div>
                  <dt>Seed / revision</dt>
                  <dd>
                    {selected.seed} / {selected.code_revision || "Not recorded"}
                  </dd>
                </div>
                <div>
                  <dt>MLflow</dt>
                  <dd>
                    <span className={`status ${selected.mlflow.status}`}>
                      {selected.mlflow.status}
                    </span>
                  </dd>
                </div>
                <div>
                  <dt>Recorded quality threshold</dt>
                  <dd>{number(selected.minimum_ssim, 4)} SSIM</dd>
                </div>
              </dl>
            </section>
            {selected.dataset_kind === "synthetic_only" && (
              <div className="notice">
                Synthetic-only experiment. These results validate the workflow;
                they do not establish performance on real or medical videos.
              </div>
            )}
            {reduction != null && (
              <div className={`notice ${reduction < 0 ? "tradeoff" : ""}`}>
                {reduction < 0
                  ? `${number(-reduction)}% higher measured bitrate`
                  : `${number(reduction)}% lower measured bitrate`}{" "}
                than fixed CRF among selected test sources. Compare measured
                quality violations and coverage before interpreting this
                difference.
              </div>
            )}
            <div className="chart-grid">
              <BarChart
                title="Measured bitrate change vs fixed CRF"
                unit="Bitrate change (%) · positive means higher bitrate"
                bars={policies.map((name) => ({
                  name: policyName(name),
                  value:
                    selected.policies[name]
                      ?.mean_bitrate_reduction_percent_vs_fixed == null
                      ? null
                      : -selected.policies[name]
                          .mean_bitrate_reduction_percent_vs_fixed!,
                }))}
              />
              <BarChart
                title="Measured quality constraint violations"
                unit="Selected test sources violating SSIM (%)"
                bars={policies.map((name) => ({
                  name: policyName(name),
                  value:
                    selected.policies[name]?.violation_rate_among_selected ==
                    null
                      ? null
                      : selected.policies[name].violation_rate_among_selected! *
                        100,
                }))}
              />
            </div>
            <section className="panel">
              <div className="panel-heading">
                <div>
                  <h3>Held-out policy comparison</h3>
                  <p>
                    Measured test outcomes · Oracle is retrospective, never an
                    inference input
                  </p>
                </div>
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Policy</th>
                      <th>Selected / test sources</th>
                      <th>Abstention</th>
                      <th>SSIM violations</th>
                      <th>Bitrate reduction vs fixed</th>
                      <th>Distance from Oracle</th>
                      <th>Oracle comparisons</th>
                    </tr>
                  </thead>
                  <tbody>
                    {policies.map((name) => {
                      const value = selected.policies[name];
                      return (
                        value && (
                          <tr key={name}>
                            <th>{policyName(name)}</th>
                            <td>
                              {value.selected_sources} / {value.test_sources}
                            </td>
                            <td>{number(value.abstention_rate * 100)}%</td>
                            <td>
                              {value.violation_rate_among_selected == null
                                ? "—"
                                : `${number(value.violation_rate_among_selected * 100)}%`}
                            </td>
                            <td>
                              {number(
                                value.mean_bitrate_reduction_percent_vs_fixed,
                              )}
                              %
                            </td>
                            <td>
                              {number(
                                value.mean_absolute_distance_from_oracle_percent,
                              )}
                              %
                            </td>
                            <td>{value.oracle_comparable_sources}</td>
                          </tr>
                        )
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="footnote">
                Negative bitrate reduction means an increase. Violation rates
                use selected sources only; policies may have different coverage.
                Oracle distance uses only sources with a feasible measured
                Oracle.
              </p>
            </section>
            <div className="chart-grid">
              <section className="panel">
                <h3>Selected-model test metrics</h3>
                <p className="muted">Frozen models, held-out source groups</p>
                <table>
                  <thead>
                    <tr>
                      <th>Prediction target</th>
                      <th>MAE</th>
                      <th>RMSE</th>
                      <th>R²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {["ssim", "bitrate_bps"].map((name) => {
                      const scores = selected.test_metrics[name];
                      return (
                        scores && (
                          <tr key={name}>
                            <th>
                              {name === "ssim" ? "SSIM" : "Bitrate (kb/s)"}
                            </th>
                            <td>
                              {number(
                                scores.mae == null
                                  ? null
                                  : scores.mae / (name === "ssim" ? 1 : 1000),
                                6,
                              )}
                            </td>
                            <td>
                              {number(
                                scores.rmse == null
                                  ? null
                                  : scores.rmse / (name === "ssim" ? 1 : 1000),
                                6,
                              )}
                            </td>
                            <td>{number(scores.r2, 4)}</td>
                          </tr>
                        )
                      );
                    })}
                  </tbody>
                </table>
                <p className="footnote">
                  A missing R² can reflect a small independent cohort or
                  constant targets; it is not zero.
                </p>
              </section>
              <section className="panel importance">
                <div className="panel-heading">
                  <div>
                    <h3>Feature sensitivity</h3>
                    <p>Validation permutation importance · not causal</p>
                  </div>
                  <label className="sr-only" htmlFor="importance-target">
                    Importance target
                  </label>
                  <select
                    id="importance-target"
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                  >
                    <option value="ssim">SSIM</option>
                    <option value="bitrate_bps">Bitrate</option>
                  </select>
                </div>
                <BarChart
                  title="Permutation importance"
                  unit={
                    target === "ssim"
                      ? "Increase in SSIM MAE"
                      : "Increase in bitrate MAE (bps)"
                  }
                  bars={Object.entries(importance).map(([name, value]) => ({
                    name:
                      name === "condition" ? "CRF" : name.replaceAll("_", " "),
                    value: value.mean,
                    note: `standard deviation ${number(value.std, 6)}`,
                  }))}
                />
                <p className="footnote">
                  Correlated features and small validation groups can make
                  importance unstable. Hover a bar for variability.
                </p>
              </section>
            </div>
            <section className="panel">
              <div className="panel-heading">
                <div>
                  <h3>Recorded CRF recommendations</h3>
                  <p>
                    Saved test-source predictions, with measured outcomes where
                    recorded
                  </p>
                </div>
                <span className="badge predicted">Predicted ≠ measured</span>
              </div>
              {!selected.recommendations.length ? (
                <State title="No published recommendations">
                  This report has no recommendations for the currently published
                  source identities.
                </State>
              ) : (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Source</th>
                        <th>Recommended CRF</th>
                        <th>Constraint status</th>
                        <th>Predicted SSIM</th>
                        <th>Measured SSIM</th>
                        <th>Predicted bitrate</th>
                        <th>Measured bitrate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.recommendations.map((row) => (
                        <tr key={row.source_name}>
                          <th>{row.source_name}</th>
                          <td>{row.crf ?? "No candidate"}</td>
                          <td>
                            {row.constraint_status === "predicted_feasible"
                              ? "Predicted feasible"
                              : "No predicted candidate"}
                          </td>
                          <td className="prediction-value">
                            {number(row.predicted_ssim, 6)}
                          </td>
                          <td>{number(row.measured_ssim, 6)}</td>
                          <td className="prediction-value">
                            {bitrate(row.predicted_bitrate_bps)}
                          </td>
                          <td>{bitrate(row.measured_bitrate_bps)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
            <section className="panel inference-panel">
              <div>
                <span className="overline">OPTIONAL INFERENCE</span>
                <h3>Ask the saved model</h3>
                <p>
                  Predict from existing cached source features. This does not
                  encode or measure a video.
                </p>
              </div>
              {selected.model_status !== "configured" ? (
                <State
                  title={
                    selected.model_status === "missing"
                      ? "Model artifact unavailable"
                      : "Live inference is disabled"
                  }
                >
                  Saved metadata and recommendations remain available. An
                  administrator must enable a trusted model for new predictions.
                </State>
              ) : (
                <form onSubmit={infer}>
                  <div className="filters">
                    <label className="wide">
                      Prediction source
                      <select
                        aria-label="Prediction source"
                        value={videoId}
                        onChange={(e) => setVideoId(e.target.value)}
                        disabled={!videos.data?.items.length}
                      >
                        <option value="" disabled>
                          Select a published source
                        </option>
                        {videos.data?.items.map((video) => (
                          <option key={video.id} value={video.id}>
                            {video.display_name}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Minimum predicted SSIM
                      <input
                        type="number"
                        min="0.000001"
                        max="1"
                        step="0.000001"
                        required
                        value={threshold}
                        onChange={(e) => setThreshold(e.target.value)}
                      />
                    </label>
                    <button
                      type="submit"
                      className="primary"
                      disabled={busy || !videoId}
                    >
                      {busy ? "Predicting…" : "Predict CRF →"}
                    </button>
                  </div>
                  <Pagination
                    offset={videoOffset}
                    limit={20}
                    total={videos.data?.total ?? 0}
                    onChange={setVideoOffset}
                    label="prediction sources"
                  />
                  {videos.error && <p role="alert">{videos.error}</p>}
                  {error && (
                    <State title="Prediction unavailable" error>
                      {error}
                    </State>
                  )}
                  {prediction && (
                    <div className="prediction-result" role="status">
                      <strong>
                        {prediction.crf == null
                          ? "No candidate satisfies the predicted quality constraint."
                          : `Recommended CRF ${prediction.crf} · predicted SSIM ${number(prediction.predicted_ssim, 6)} · ${bitrate(prediction.predicted_bitrate_bps)}`}
                      </strong>
                      <p>{prediction.explanation}</p>
                    </div>
                  )}
                </form>
              )}
            </section>
          </>
        )
      )}
    </>
  );
}
