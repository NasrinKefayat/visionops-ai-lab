import { useState } from "react";
import { number, State } from "./components";

export type Point = {
  id: string;
  x: number;
  y: number;
  label: string;
  group: string;
};
const colors = ["#087f8c", "#6653b5", "#ba6814", "#ba4667", "#366eae"];
export function ScatterChart({
  title,
  points,
  xLabel,
  yLabel,
  selected,
  onSelect,
}: {
  title: string;
  points: Point[];
  xLabel: string;
  yLabel: string;
  selected?: string;
  onSelect?: (id: string) => void;
}) {
  const [hover, setHover] = useState<Point | null>(null);
  const valid = points.filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.y),
  );
  if (!valid.length)
    return (
      <section className="panel chart-panel">
        <h3>{title}</h3>
        <State title="No finite measurements">
          Metrics will appear when a matching encode has been evaluated.
        </State>
      </section>
    );
  const range = (values: number[], lower = -Infinity, upper = Infinity) => {
    const min = Math.min(...values),
      max = Math.max(...values);
    const pad = Math.max((max - min) * 0.12, Math.abs(max) * 0.01, 0.001);
    return [Math.max(lower, min - pad), Math.min(upper, max + pad)];
  };
  const integerX = xLabel === "CRF";
  let [xmin, xmax] = range(
    valid.map((p) => p.x),
    0,
  );
  if (integerX) {
    xmin = Math.max(0, Math.floor(xmin));
    xmax = Math.min(51, Math.ceil(xmax));
  }
  const [ymin, ymax] = yLabel.includes("SSIM")
    ? range(
        valid.map((p) => p.y),
        -1,
        1,
      )
    : range(
        valid.map((p) => p.y),
        yLabel.includes("PSNR") ? -Infinity : 0,
      );
  const sx = (x: number) => 70 + ((x - xmin) / (xmax - xmin)) * 525;
  const sy = (y: number) => 232 - ((y - ymin) / (ymax - ymin)) * 195;
  const groups = [...new Set(valid.map((p) => p.group))].sort();
  return (
    <figure className="panel chart-panel">
      <figcaption>
        <h3>{title}</h3>
        <span className="measure-label">Measured · selected source</span>
      </figcaption>
      <svg
        viewBox="0 0 640 300"
        role="img"
        aria-label={`${title}. ${xLabel} versus ${yLabel}.`}
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1="70"
              x2="595"
              y1={sy(ymin + t * (ymax - ymin))}
              y2={sy(ymin + t * (ymax - ymin))}
              className="grid-line"
            />
            <text x="60" y={sy(ymin + t * (ymax - ymin)) + 4} textAnchor="end">
              {number(ymin + t * (ymax - ymin), 4)}
            </text>
            <text
              x={sx(
                integerX
                  ? Math.round(xmin + t * (xmax - xmin))
                  : xmin + t * (xmax - xmin),
              )}
              y="251"
              textAnchor="middle"
            >
              {number(
                integerX
                  ? Math.round(xmin + t * (xmax - xmin))
                  : xmin + t * (xmax - xmin),
                2,
              )}
            </text>
          </g>
        ))}
        <line x1="70" x2="595" y1="232" y2="232" className="axis-line" />
        <text x="334" y="285" textAnchor="middle" className="axis-label">
          {xLabel}
        </text>
        <text
          transform="translate(18 136) rotate(-90)"
          textAnchor="middle"
          className="axis-label"
        >
          {yLabel}
        </text>
        {valid.map((p) => (
          <circle
            key={p.id}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={selected === p.id ? 8 : 5.5}
            fill={colors[groups.indexOf(p.group) % colors.length]}
            stroke={selected === p.id ? "#172b40" : "white"}
            strokeWidth="2"
            tabIndex={0}
            role="button"
            aria-label={p.label}
            onMouseEnter={() => setHover(p)}
            onMouseLeave={() => setHover(null)}
            onFocus={() => setHover(p)}
            onBlur={() => setHover(null)}
            onClick={() => onSelect?.(p.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect?.(p.id);
              }
            }}
          >
            <title>{p.label}</title>
          </circle>
        ))}
      </svg>
      <div className="chart-tooltip" aria-live="polite">
        {hover?.label ||
          "Hover or focus a point for details. Select to inspect the encoding."}
      </div>
      <div className="legend">
        {groups.map((group, i) => (
          <span key={group}>
            <i style={{ background: colors[i % colors.length] }} />
            {group}
          </span>
        ))}
      </div>
    </figure>
  );
}
function barValue(value: number | null) {
  return value != null && value !== 0 && Math.abs(value) < 0.001
    ? value.toExponential(2)
    : number(value, 3);
}

export function BarChart({
  title,
  bars,
  unit,
}: {
  title: string;
  bars: { name: string; value: number | null; note?: string }[];
  unit: string;
}) {
  const valid = bars.filter((b) => b.value != null && Number.isFinite(b.value));
  if (!valid.length)
    return (
      <section className="panel chart-panel">
        <h3>{title}</h3>
        <State title="No comparison available">
          This experiment has no comparable measured results.
        </State>
      </section>
    );
  const min = Math.min(0, ...valid.map((b) => b.value!)),
    max = Math.max(0, ...valid.map((b) => b.value!));
  const span = max - min || 1;
  const sx = (value: number) => 180 + ((value - min) / span) * 370;
  return (
    <figure className="panel chart-panel">
      <figcaption>
        <h3>{title}</h3>
        <span className="measure-label">{unit}</span>
      </figcaption>
      <svg
        viewBox={`0 0 640 ${Math.max(130, bars.length * 42 + 65)}`}
        role="img"
        aria-label={title}
      >
        <line
          x1={sx(0)}
          x2={sx(0)}
          y1="12"
          y2={bars.length * 42 + 12}
          className="axis-line"
        />
        {bars.map((bar, i) => (
          <g key={bar.name}>
            <text x="168" y={i * 42 + 32} textAnchor="end">
              {bar.name}
            </text>
            {bar.value != null && (
              <rect
                x={Math.min(sx(0), sx(bar.value))}
                y={i * 42 + 16}
                width={Math.max(1, Math.abs(sx(bar.value) - sx(0)))}
                height="23"
                rx="3"
                fill={colors[i % colors.length]}
              >
                <title>
                  {bar.name}: {number(bar.value, 4)} {unit}. {bar.note}
                </title>
              </rect>
            )}
            <text x="565" y={i * 42 + 32}>
              {barValue(bar.value)}
            </text>
          </g>
        ))}
        <text
          x="365"
          y={bars.length * 42 + 55}
          textAnchor="middle"
          className="axis-label"
        >
          {unit}
        </text>
      </svg>
    </figure>
  );
}
