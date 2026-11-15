import { useEffect, useState } from "react";
import { query, useRemote } from "./api";
import {
  bitrate,
  fileSize,
  KindBadge,
  number,
  Pagination,
  shortId,
  State,
} from "./components";
import { ScatterChart, type Point } from "./charts";
import type { Encoding, Page, Video } from "./types";

export function QualityView({ refresh }: { refresh: number }) {
  const [dataset, setDataset] = useState("");
  const [search, setSearch] = useState("");
  const [videoOffset, setVideoOffset] = useState(0);
  const [sourceId, setSourceId] = useState<number | null>(null);
  const [crf, setCrf] = useState("");
  const [run, setRun] = useState("");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const videos = useRemote<Page<Video>>(
    query("videos", { dataset, search, offset: videoOffset, limit: 20 }),
    refresh,
  );
  useEffect(() => {
    if (videos.data && !videos.data.items.some((v) => v.id === sourceId))
      setSourceId(videos.data.items[0]?.id ?? null);
  }, [videos.data, sourceId]);
  useEffect(() => {
    setCrf("");
    setRun("");
    setOffset(0);
    setSelected(null);
  }, [sourceId]);
  const video = useRemote<Video>(
    sourceId ? `/api/v1/videos/${sourceId}` : null,
    refresh,
  );
  const encodings = useRemote<Page<Encoding>>(
    sourceId
      ? query(`videos/${sourceId}/quality`, {
          crf,
          run_id: run,
          offset,
          limit: 20,
        })
      : null,
    refresh,
  );
  const rows = encodings.data?.items || [];
  const experiment = useRemote<Encoding>(
    selected ? `/api/v1/experiments/${selected}` : null,
    refresh,
  );
  const points = (
    x: (r: Encoding) => number | null,
    y: (r: Encoding) => number | null,
  ): Point[] =>
    rows.flatMap((row) => {
      const xv = x(row),
        yv = y(row);
      return xv == null || yv == null
        ? []
        : [
            {
              id: row.experiment_id,
              x: xv,
              y: yv,
              group: `CRF ${row.crf}`,
              label: `CRF ${row.crf} · ${row.preset} · ${bitrate(row.output_bitrate_bps)} · SSIM ${number(row.measured_ssim, 6)} · run ${shortId(row.run_id)}`,
            },
          ];
    });
  return (
    <>
      <div className="section-heading">
        <div>
          <p className="eyebrow">01 / VIDEO QUALITY</p>
          <h2>Every encode, in perspective.</h2>
          <p>
            Compare measured quality and compression for one source at a time.
          </p>
        </div>
        <span className="badge measured">Measured results</span>
      </div>
      <section className="panel filters" aria-label="Video filters">
        <label>
          Dataset
          <select
            aria-label="Dataset"
            value={dataset}
            onChange={(e) => {
              setDataset(e.target.value);
              setVideoOffset(0);
            }}
          >
            <option value="">All published sources</option>
            <option value="synthetic_video">Synthetic video</option>
            <option value="public_video">Public video</option>
          </select>
        </label>
        <label>
          Find a source
          <input
            placeholder="Search source name"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setVideoOffset(0);
            }}
          />
        </label>
        <label className="wide">
          Source video
          <select
            aria-label="Source video"
            value={sourceId ?? ""}
            onChange={(e) => setSourceId(Number(e.target.value))}
            disabled={!videos.data?.items.length}
          >
            <option value="" disabled>
              Select a video
            </option>
            {videos.data?.items.map((v) => (
              <option key={v.id} value={v.id}>
                {v.display_name} · {v.width} × {v.height}
              </option>
            ))}
          </select>
        </label>
        <Pagination
          offset={videoOffset}
          limit={20}
          total={videos.data?.total ?? 0}
          onChange={setVideoOffset}
          label="videos"
        />
      </section>
      {videos.loading ? (
        <State title="Loading sources…" />
      ) : videos.error ? (
        <State title="Sources could not be loaded" error>
          {videos.error}
        </State>
      ) : !videos.data?.items.length ? (
        <State title="No published videos yet">
          Run the documented experiment setup and publish a nonsensitive dataset
          to see measured results here.
        </State>
      ) : (
        <>
          {video.error && (
            <State title="Video unavailable" error>
              {video.error}
            </State>
          )}
          {video.data && (
            <section className="panel source-summary">
              <div>
                <span className="overline">SELECTED SOURCE</span>
                <h3>{video.data.display_name}</h3>
                <KindBadge kind={video.data.dataset_kind} />
              </div>
              <dl>
                <div>
                  <dt>Resolution</dt>
                  <dd>
                    {video.data.width} × {video.data.height}
                  </dd>
                </div>
                <div>
                  <dt>Duration / FPS</dt>
                  <dd>
                    {number(video.data.duration_seconds)} s /{" "}
                    {number(video.data.fps)}
                  </dd>
                </div>
                <div>
                  <dt>Source bitrate</dt>
                  <dd>{bitrate(video.data.bitrate_bps)}</dd>
                </div>
                <div>
                  <dt>Source file / codec</dt>
                  <dd>
                    {fileSize(video.data.file_size_bytes)} / {video.data.codec}
                  </dd>
                </div>
              </dl>
            </section>
          )}
          <div className="inline-filters">
            <label>
              CRF
              <select
                aria-label="CRF"
                value={crf}
                onChange={(e) => {
                  setCrf(e.target.value);
                  setOffset(0);
                  setSelected(null);
                }}
              >
                <option value="">All CRFs</option>
                {video.data?.available_crfs.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Encoding run
              <select
                aria-label="Encoding run"
                value={run}
                onChange={(e) => {
                  setRun(e.target.value);
                  setOffset(0);
                  setSelected(null);
                }}
              >
                <option value="">All runs for this source</option>
                {video.data?.available_run_ids?.map((id) => (
                  <option key={id} value={id}>
                    Run {shortId(id)}
                  </option>
                ))}
              </select>
            </label>
            <span className="muted">
              Charts use only this source and the current result page.
            </span>
          </div>
          {encodings.loading ? (
            <State title="Loading measurements…" />
          ) : encodings.error ? (
            <State title="Measurements unavailable" error>
              {encodings.error}
            </State>
          ) : !rows.length ? (
            <State title="No matching encodes">
              Try another CRF or run, or complete the Phase 2 baseline for this
              source.
            </State>
          ) : (
            <>
              <div className="chart-grid">
                <ScatterChart
                  title="Bitrate versus SSIM"
                  xLabel="Output bitrate (kb/s)"
                  yLabel="Measured SSIM"
                  points={points(
                    (r) =>
                      r.output_bitrate_bps == null
                        ? null
                        : r.output_bitrate_bps / 1000,
                    (r) => r.measured_ssim,
                  )}
                  selected={selected ?? undefined}
                  onSelect={setSelected}
                />
                <ScatterChart
                  title="CRF versus PSNR"
                  xLabel="CRF"
                  yLabel="Measured PSNR (dB)"
                  points={points(
                    (r) => r.crf,
                    (r) => r.measured_psnr_db,
                  )}
                  selected={selected ?? undefined}
                  onSelect={setSelected}
                />
                <ScatterChart
                  title="CRF versus bitrate"
                  xLabel="CRF"
                  yLabel="Output bitrate (kb/s)"
                  points={points(
                    (r) => r.crf,
                    (r) =>
                      r.output_bitrate_bps == null
                        ? null
                        : r.output_bitrate_bps / 1000,
                  )}
                  selected={selected ?? undefined}
                  onSelect={setSelected}
                />
                <ScatterChart
                  title="Encoding duration"
                  xLabel="CRF"
                  yLabel="Encoding wall time (s)"
                  points={points(
                    (r) => r.crf,
                    (r) => r.encoding_seconds,
                  )}
                  selected={selected ?? undefined}
                  onSelect={setSelected}
                />
              </div>
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h3>CRF comparison</h3>
                    <p>Measured outputs · select an encoding for details</p>
                  </div>
                  <span className="badge">{encodings.data?.total} records</span>
                </div>
                <div className="table-scroll">
                  <table>
                    <caption className="sr-only">
                      Measured encoding results for the selected video
                    </caption>
                    <thead>
                      <tr>
                        <th>Experiment</th>
                        <th>CRF / preset</th>
                        <th>Status</th>
                        <th>SSIM</th>
                        <th>PSNR</th>
                        <th>Bitrate</th>
                        <th>Size</th>
                        <th>Ratio</th>
                        <th>Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row) => (
                        <tr
                          key={row.experiment_id}
                          className={
                            selected === row.experiment_id ? "selected" : ""
                          }
                        >
                          <td>
                            <button
                              className="text-button"
                              onClick={() => setSelected(row.experiment_id)}
                            >
                              {shortId(row.experiment_id)}
                            </button>
                          </td>
                          <td>
                            {row.crf} / {row.preset}
                          </td>
                          <td>
                            <span className={`status ${row.status}`}>
                              {row.status}
                            </span>
                          </td>
                          <td>{number(row.measured_ssim, 6)}</td>
                          <td>
                            {row.psnr_infinite
                              ? "∞"
                              : number(row.measured_psnr_db)}{" "}
                            dB
                          </td>
                          <td>{bitrate(row.output_bitrate_bps)}</td>
                          <td>{fileSize(row.output_file_size_bytes)}</td>
                          <td>{number(row.compression_ratio)}×</td>
                          <td>{number(row.encoding_seconds, 3)} s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <Pagination
                  offset={offset}
                  limit={20}
                  total={encodings.data?.total ?? 0}
                  onChange={(value) => {
                    setOffset(value);
                    setSelected(null);
                  }}
                  label="encodings"
                />
                <p className="footnote">
                  Compression ratio = source bytes / output bytes. Container
                  bitrates include overhead; encoding drops audio. Infinite PSNR
                  is shown in the table and excluded from finite charts. Missing
                  values are shown as —.
                </p>
              </section>
              {selected && (
                <section className="panel">
                  <h3>Encoding detail · {shortId(selected)}</h3>
                  {experiment.loading ? (
                    <p role="status">Loading experiment…</p>
                  ) : experiment.error ? (
                    <p role="alert">{experiment.error}</p>
                  ) : (
                    experiment.data && (
                      <dl className="detail-grid">
                        <div>
                          <dt>Run</dt>
                          <dd>{shortId(experiment.data.run_id)}</dd>
                        </div>
                        <div>
                          <dt>Seed</dt>
                          <dd>{experiment.data.seed}</dd>
                        </div>
                        <div>
                          <dt>Code revision</dt>
                          <dd>
                            {experiment.data.code_revision || "Not recorded"}
                          </dd>
                        </div>
                        <div>
                          <dt>Bitrate reduction vs source</dt>
                          <dd>
                            {number(experiment.data.bitrate_reduction_percent)}%
                          </dd>
                        </div>
                        <div>
                          <dt>Processing</dt>
                          <dd>
                            {experiment.data.error_code
                              ? "Encoding or evaluation failed"
                              : experiment.data.status}
                          </dd>
                        </div>
                      </dl>
                    )
                  )}
                </section>
              )}
            </>
          )}
        </>
      )}
    </>
  );
}
