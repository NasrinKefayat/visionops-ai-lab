import type { ReactNode } from "react";

export function number(value: number | null | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(
        Object.is(value, -0) ? 0 : value,
      );
}
export const bitrate = (value: number | null | undefined) =>
  value == null ? "—" : `${number(value / 1000, 2)} kb/s`;
export const fileSize = (value: number | null | undefined) =>
  value == null ? "—" : `${number(value / 1024, 1)} KiB`;
export const shortId = (id: string) => id.slice(0, 8);
export function KindBadge({ kind }: { kind: string }) {
  return (
    <span
      className={`badge ${kind.startsWith("synthetic") ? "synthetic" : "public"}`}
    >
      {kind.startsWith("synthetic") ? "Synthetic data" : "Public video data"}
    </span>
  );
}
export function State({
  title,
  children,
  error = false,
}: {
  title: string;
  children?: ReactNode;
  error?: boolean;
}) {
  return (
    <div
      className={`state ${error ? "error" : ""}`}
      role={error ? "alert" : "status"}
    >
      <span className="state-symbol" aria-hidden="true">
        {error ? "!" : "◇"}
      </span>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
    </div>
  );
}
export function KPI({
  label,
  value,
  note,
  accent = false,
}: {
  label: string;
  value: string;
  note?: string;
  accent?: boolean;
}) {
  return (
    <article className={`kpi ${accent ? "accent" : ""}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      {note && <span>{note}</span>}
    </article>
  );
}
export function Pagination({
  offset,
  limit,
  total,
  onChange,
  label,
}: {
  offset: number;
  limit: number;
  total: number;
  onChange: (offset: number) => void;
  label: string;
}) {
  return (
    <div className="pagination">
      <span>
        {total
          ? `${offset + 1}–${Math.min(offset + limit, total)} of ${total}`
          : "0 results"}
      </span>
      <div>
        <button
          type="button"
          aria-label={`Previous ${label}`}
          disabled={offset === 0}
          onClick={() => onChange(Math.max(0, offset - limit))}
        >
          ← Previous
        </button>
        <button
          type="button"
          aria-label={`Next ${label}`}
          disabled={offset + limit >= total}
          onClick={() => onChange(offset + limit)}
        >
          Next →
        </button>
      </div>
    </div>
  );
}
