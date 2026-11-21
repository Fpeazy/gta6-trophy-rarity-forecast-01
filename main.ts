// gta6-trophy-rarity-forecast-01 — counts-model forecast (placeholder v1)
// Forecasts each GTA VI (PS5) trophy's Exophase earn rate at the case target.
//
// Method (v1, counts model):
//   * From every history read, recover earners E = rate * players / 100 and
//     players tracked P. Both are modelled separately in log space against
//     x = ln(1 + hours since 2026-11-19T00:00:00Z) by least squares, so the
//     rate is E(t)/P(t): earners growing against a saturating player base.
//     With < 3 usable reads (or a flat history, as on launch night) the fit
//     is degenerate and we fall back to the last read's rate.
//   * Spread on the logit scale: a random walk sigma_rw = 0.5 * sqrt(days)
//     combined (in quadrature) with the fit residual, so a flat history still
//     carries an honest, widening band instead of a point forecast.
//   * Deterministic: no randomness anywhere, fixed z-scores, fixed rounding.
//   * Node 22, standard library only, no network.
//
// Written from scratch from RULES.txt and the .in case schema; no starter
// code was read or copied.
import { readdirSync, readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const EPOCH_MS = Date.parse("2026-11-19T00:00:00Z");
const MS_PER_HOUR = 3_600_000;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function logit(pct: number): number {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}

function invLogit(l: number): number {
  return (100 / (1 + Math.exp(-l)));
}

function round4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

// Least squares y = a + b*x. Returns null when x has no variance.
function linFit(xs: number[], ys: number[]): { a: number; b: number; residSd: number } | null {
  const n = xs.length;
  if (n < 3) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx < 1e-12) return null;
  const b = sxy / sxx;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    ss += r * r;
  }
  const residSd = n > 2 ? Math.sqrt(ss / (n - 2)) : 0;
  return { a, b, residSd };
}

interface Trophy {
  id: number;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: (number | null)[];
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

function forecastCase(c: CaseFile): { trophies: { id: number; q: number[] }[] } {
  const targetMs = Date.parse(c.target);
  const horizonDays = c.horizon_minutes / 1440;
  const xTarget = Math.log(1 + (targetMs - EPOCH_MS) / MS_PER_HOUR);
  const reads = c.history;
  const last = reads[reads.length - 1];
  const nT = c.trophies.length;
  // No history at all (defensive; the practice/final cases always carry
  // reads): fall back to a flat 1% median with a wide honest band.
  if (!last) {
    return {
      trophies: c.trophies.map((t) => ({
        id: t.id,
        q: Z.map((z) => round4(clamp(invLogit(logit(1) + z * 2.5), 0.01, 100))),
      })),
    };
  }

  // Players-tracked growth: log P vs x (power-law / saturating in log space).
  const px: number[] = [];
  const py: number[] = [];
  for (const r of reads) {
    if (typeof r.players_tracked === "number" && r.players_tracked > 0) {
      px.push(Math.log(1 + (Date.parse(r.at) - EPOCH_MS) / MS_PER_HOUR));
      py.push(Math.log(r.players_tracked));
    }
  }
  const pFit = linFit(px, py);
  const pSlope = pFit ? clamp(pFit.b, 0, 2) : 0;
  const pTarget = pFit
    ? Math.exp(pFit.a + pSlope * xTarget)
    : last.players_tracked;
  const pResid = pFit ? pFit.residSd : 0;

  const rows: { id: number; q: number[] }[] = [];
  for (let t = 0; t < nT; t++) {
    const lastRate = last.rates[t];
    // Trophy pinned at (or effectively at) 100%: it cannot go above 100.
    if (typeof lastRate === "number" && lastRate >= 99.99) {
      rows.push({ id: c.trophies[t].id, q: [99.95, 99.99, 100, 100, 100] });
      continue;
    }
    // Earners growth: log(E + 0.5) vs x (the +0.5 keeps zero earners finite).
    const ex: number[] = [];
    const ey: number[] = [];
    for (const r of reads) {
      const rate = r.rates[t];
      if (typeof rate === "number" && typeof r.players_tracked === "number") {
        ex.push(Math.log(1 + (Date.parse(r.at) - EPOCH_MS) / MS_PER_HOUR));
        ey.push(Math.log((rate * r.players_tracked) / 100 + 0.5));
      }
    }
    const eFit = linFit(ex, ey);
    let median: number;
    let resid = pResid;
    if (eFit && typeof lastRate === "number") {
      const eSlope = clamp(eFit.b, -3, 3);
      const eTarget = Math.exp(eFit.a + eSlope * xTarget) - 0.5;
      median = clamp((100 * Math.max(eTarget, 0)) / Math.max(pTarget, 1), 0.01, 99.99);
      resid = Math.sqrt(eFit.residSd * eFit.residSd + pResid * pResid);
    } else {
      median = typeof lastRate === "number" ? clamp(lastRate, 0.01, 99.99) : 0.01;
    }
    // Spread: random walk 0.5*sqrt(days) combined with fit residual.
    const sigma = clamp(Math.sqrt(0.25 * horizonDays + resid * resid), 0.1, 2.5);
    const m = logit(median);
    const q = Z.map((z) => round4(clamp(invLogit(m + z * sigma), 0.01, 100)));
    // Enforce non-decreasing after rounding.
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
    rows.push({ id: c.trophies[t].id, q });
  }
  return { trophies: rows };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output/";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const c = JSON.parse(readFileSync(join(inputDir, f), "utf8")) as CaseFile;
    const out = forecastCase(c);
    const text = JSON.stringify(out);
    if (text.length > 65536) throw new Error(`output for ${c.id} exceeds 65536 bytes`);
    writeFileSync(join(outputDir, `${c.id}.out`), text + "\n", "utf8");
  }
}

main();
