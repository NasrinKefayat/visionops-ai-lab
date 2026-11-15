export type Page<T> = {
  items: T[];
  total: number;
  offset: number;
  limit: number;
  available: boolean;
  message: string | null;
};
export type Video = {
  id: number;
  display_name: string;
  dataset_kind: "synthetic_video" | "public_video";
  duration_seconds: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  bitrate_bps: number;
  file_size_bytes: number;
  available_crfs: number[];
  available_run_ids: string[];
};
export type Encoding = {
  experiment_id: string;
  run_id: string;
  video_id: number;
  crf: number;
  preset: string;
  status: string;
  width: number;
  height: number;
  output_bitrate_bps: number | null;
  output_file_size_bytes: number | null;
  encoding_seconds: number | null;
  compression_ratio: number | null;
  bitrate_reduction_percent: number | null;
  measured_ssim: number | null;
  measured_psnr_db: number | null;
  psnr_infinite: boolean;
  started_at: string;
  completed_at: string | null;
  seed: number;
  code_revision: string | null;
  error_code: string | null;
};
export type Monitoring = {
  database_status: "ready" | "unavailable";
  publication_status: string;
  source_videos: number | null;
  completed_encodes: number | null;
  average_measured_ssim: number | null;
  average_measured_psnr_db: number | null;
  infinite_psnr_count: number | null;
  average_output_bitrate_bps: number | null;
  average_encoding_seconds: number | null;
  available_ml_experiments: number;
  status_counts: Record<string, number>;
  message: string | null;
};
export type Recommendation = {
  source_name: string;
  crf: number | null;
  predicted_ssim: number | null;
  predicted_bitrate_bps: number | null;
  constraint_status: "predicted_feasible" | "no_predicted_candidate";
  explanation: string;
  measured_ssim: number | null;
  measured_bitrate_bps: number | null;
};
export type Policy = {
  test_sources: number;
  selected_sources: number;
  abstention_rate: number;
  violation_rate_among_selected: number | null;
  mean_bitrate_reduction_percent_vs_fixed: number | null;
  mean_absolute_distance_from_oracle_percent: number | null;
  oracle_comparable_sources: number;
};
export type StudySummary = {
  id: string;
  dataset_kind: "synthetic_only" | "public_video";
  source_count: number;
  row_count: number;
  seed: number;
  code_revision: string | null;
  selected_models: Record<string, string>;
  extractor_version: string;
  model_status: "missing" | "metadata_only" | "configured";
  minimum_ssim: number | null;
  mlflow: { run_id: string | null; status: string };
};
export type Study = StudySummary & {
  split_groups: Record<string, number>;
  test_metrics: Record<
    string,
    { mae: number | null; rmse: number | null; r2: number | null }
  >;
  policies: Record<string, Policy>;
  feature_importance: Record<
    string,
    Record<string, { mean: number; std: number }>
  >;
  recommendations: Recommendation[];
};
