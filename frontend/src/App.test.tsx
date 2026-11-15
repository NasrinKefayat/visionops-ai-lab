import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { number } from "./components";
import { ScatterChart } from "./charts";
import type { Encoding, Monitoring, Page, Study, Video } from "./types";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
const page = <T,>(items: T[]): Page<T> => ({
  items,
  total: items.length,
  offset: 0,
  limit: 20,
  available: true,
  message: null,
});
const videos: Video[] = [1, 2].map((id) => ({
  id,
  display_name: `test-source-${id}`,
  dataset_kind: "synthetic_video",
  duration_seconds: 2,
  width: 320,
  height: 240,
  fps: 24,
  codec: "h264",
  bitrate_bps: 100000,
  file_size_bytes: 20000,
  available_crfs: [18, 23],
  available_run_ids: ["test-run"],
}));
const encodings: Encoding[] = [18, 23].map((crf, i) => ({
  experiment_id: `aaaa111${i}-test`,
  run_id: "test-run",
  video_id: 1,
  crf,
  preset: "medium",
  status: "completed",
  width: 320,
  height: 240,
  output_bitrate_bps: 50000 - i * 10000,
  output_file_size_bytes: 5000,
  encoding_seconds: 0.2,
  compression_ratio: 4,
  bitrate_reduction_percent: 50,
  measured_ssim: 0.99 - i * 0.01,
  measured_psnr_db: 40,
  psnr_infinite: false,
  started_at: "2026-01-01T00:00:00",
  completed_at: "2026-01-01T00:00:01",
  seed: 42,
  code_revision: "abcdef0",
  error_code: null,
}));
const monitor: Monitoring = {
  database_status: "ready",
  publication_status: "ready",
  source_videos: 2,
  completed_encodes: 2,
  average_measured_ssim: 0.985,
  average_measured_psnr_db: 40,
  infinite_psnr_count: 0,
  average_output_bitrate_bps: 45000,
  average_encoding_seconds: 0.2,
  available_ml_experiments: 1,
  status_counts: { completed: 2 },
  message: null,
};
const policy = {
  test_sources: 2,
  selected_sources: 2,
  abstention_rate: 0,
  violation_rate_among_selected: 0,
  mean_bitrate_reduction_percent_vs_fixed: -25,
  mean_absolute_distance_from_oracle_percent: 5,
  oracle_comparable_sources: 2,
};
const study: Study = {
  id: "test-study",
  dataset_kind: "synthetic_only",
  source_count: 6,
  row_count: 18,
  seed: 42,
  code_revision: "abcdef0",
  selected_models: { ssim: "random_forest", bitrate_bps: "dummy" },
  extractor_version: "opencv-gray-v1",
  model_status: "metadata_only",
  minimum_ssim: 0.99,
  mlflow: { run_id: null, status: "unavailable" },
  split_groups: { train: 3, validation: 1, test: 2 },
  test_metrics: {
    ssim: { mae: 0.01, rmse: 0.02, r2: null },
    bitrate_bps: { mae: 1000, rmse: 2000, r2: null },
  },
  policies: {
    fixed: {
      ...policy,
      mean_bitrate_reduction_percent_vs_fixed: 0,
      violation_rate_among_selected: 0.5,
    },
    content_aware: policy,
    oracle: { ...policy, mean_absolute_distance_from_oracle_percent: 0 },
  },
  feature_importance: { ssim: { condition: { mean: 0.01, std: 0.002 } } },
  recommendations: [
    {
      source_name: "test-source-1",
      crf: 18,
      predicted_ssim: 0.995,
      predicted_bitrate_bps: 50000,
      constraint_status: "predicted_feasible",
      explanation: "Measurement required.",
      measured_ssim: 0.99,
      measured_bitrate_bps: 52000,
    },
  ],
};

function mockAPI({ empty = false, configured = false } = {}) {
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), "http://localhost");
    let data: unknown;
    if (init?.method === "POST")
      data = {
        source_name: "test-source-1",
        crf: null,
        predicted_ssim: null,
        predicted_bitrate_bps: null,
        constraint_status: "no_predicted_candidate",
        explanation:
          "No candidate is predicted to meet the quality constraint.",
        measured_ssim: null,
        measured_bitrate_bps: null,
      };
    else if (url.pathname === "/api/v1/monitoring")
      data = empty
        ? {
            ...monitor,
            database_status: "unavailable",
            publication_status: "not_configured",
            source_videos: null,
            completed_encodes: null,
            average_measured_ssim: null,
            average_measured_psnr_db: null,
            average_output_bitrate_bps: null,
            average_encoding_seconds: null,
            available_ml_experiments: 0,
            status_counts: {},
          }
        : monitor;
    else if (url.pathname === "/api/v1/videos")
      data = page(
        empty
          ? []
          : videos.filter(
              (v) =>
                !url.searchParams.get("search") ||
                v.display_name.includes(url.searchParams.get("search")!),
            ),
      );
    else if (url.pathname.match(/^\/api\/v1\/videos\/\d+$/))
      data = videos.find((v) => v.id === Number(url.pathname.split("/").pop()));
    else if (url.pathname.endsWith("/quality"))
      data = page(
        encodings
          .filter(
            (e) =>
              !url.searchParams.get("crf") ||
              String(e.crf) === url.searchParams.get("crf"),
          )
          .map((e) => ({ ...e, video_id: Number(url.pathname.split("/")[4]) })),
      );
    else if (url.pathname.startsWith("/api/v1/experiments/"))
      data = encodings.find(
        (e) => e.experiment_id === url.pathname.split("/").pop(),
      );
    else if (url.pathname === "/api/v1/studies")
      data = page(empty ? [] : [study]);
    else if (url.pathname.startsWith("/api/v1/studies/"))
      data = {
        ...study,
        model_status: configured ? "configured" : "metadata_only",
      };
    else
      return new Response(JSON.stringify({ error: { message: "Not found" } }), {
        status: 404,
      });
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

describe("dashboard data states", () => {
  it("shows empty states and missing KPIs without fabricated zero metrics", async () => {
    mockAPI({ empty: true });
    render(<App />);
    await screen.findByText("No published videos yet");
    expect(
      screen.getByText("Average measured SSIM").closest("article")!.textContent,
    ).toContain("—");
    expect(
      screen.queryByRole("img", { name: /Bitrate versus SSIM/ }),
    ).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: /Optimization/ }));
    await screen.findByText("No published ML experiments yet");
  });
  it("renders measured charts, CRF filtering, and encoding detail from backend data", async () => {
    const fetchMock = mockAPI();
    render(<App />);
    await screen.findByRole("img", { name: /Bitrate versus SSIM/ });
    expect(
      within(
        screen.getByRole("table", {
          name: "Measured encoding results for the selected video",
        }),
      ).getAllByRole("row"),
    ).toHaveLength(3);
    await userEvent.selectOptions(screen.getByLabelText("CRF"), "23");
    await waitFor(() =>
      expect(
        within(
          screen.getByRole("table", {
            name: "Measured encoding results for the selected video",
          }),
        ).getAllByRole("row"),
      ).toHaveLength(2),
    );
    expect(
      fetchMock.mock.calls.some(([input]) => String(input).includes("crf=23")),
    ).toBe(true);
    await userEvent.click(screen.getByRole("button", { name: "aaaa1111" }));
    await screen.findByText("Encoding detail · aaaa1111");
    await waitFor(() => expect(screen.getByText("abcdef0")).toBeTruthy());
  });
  it("distinguishes predictions, synthetic data, and measured bitrate increases", async () => {
    mockAPI();
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /Optimization/ }));
    await screen.findByText("Recorded CRF recommendations");
    expect(screen.getByText(/25% higher measured bitrate/)).toBeTruthy();
    expect(screen.getByText(/Synthetic-only experiment/)).toBeTruthy();
    expect(screen.getByText("Live inference is disabled")).toBeTruthy();
    expect(
      screen.getByRole("columnheader", { name: "Predicted SSIM" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("columnheader", { name: "Measured SSIM" }),
    ).toBeTruthy();
  });
  it("reports explicit abstention from enabled inference", async () => {
    const fetchMock = mockAPI({ configured: true });
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /Optimization/ }));
    const submit = await screen.findByRole("button", { name: "Predict CRF →" });
    fireEvent.change(screen.getByLabelText("Minimum predicted SSIM"), {
      target: { value: "1" },
    });
    await userEvent.click(submit);
    await screen.findByText(
      "No candidate satisfies the predicted quality constraint.",
    );
    const call = fetchMock.mock.calls.find(
      ([, init]) => init?.method === "POST",
    );
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({
      video_id: 1,
      minimum_ssim: 1,
    });
  });
  it("shows loading and recoverable backend errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => new Promise(() => {})),
    );
    const initial = render(<App />);
    expect(screen.getByText("Loading sources…")).toBeTruthy();
    initial.unmount();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Backend connection unavailable")),
    );
    render(<App />);
    await screen.findByText("Sources could not be loaded");
    mockAPI({ empty: true });
    await userEvent.click(
      screen.getByRole("button", { name: "Refresh dashboard data" }),
    );
    await screen.findByText("No published videos yet");
  });
});
it("keeps numeric zero distinct from missing or invalid metrics", () => {
  expect(number(0)).toBe("0");
  expect(number(null)).toBe("—");
  expect(number(Number.NaN)).toBe("—");
});
it("supports keyboard-accessible chart inspection", () => {
  const select = vi.fn();
  render(
    <ScatterChart
      title="Test measured chart"
      xLabel="kb/s"
      yLabel="SSIM"
      points={[
        {
          id: "point-1",
          x: 10,
          y: 0.99,
          label: "Measured CRF 18",
          group: "CRF 18",
        },
      ]}
      onSelect={select}
    />,
  );
  const point = screen.getByRole("button", { name: "Measured CRF 18" });
  fireEvent.focus(point);
  fireEvent.keyDown(point, { key: "Enter" });
  expect(select).toHaveBeenCalledWith("point-1");
  expect(screen.getAllByText("Measured CRF 18").length).toBeGreaterThan(0);
});
