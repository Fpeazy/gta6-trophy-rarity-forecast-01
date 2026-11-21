// DualCurve_RefreshDilution_v1
// GTA VI trophy rarity forecast — make.fun gta6-trophy-rarity-forecast-01.
// Node 22, standard library only, no network, fully deterministic.
//
// Method (synthesis of the team decision: counts model + refresh/dilution
// inversion). Exophase prints each trophy's earn rate to 2 decimals and a
// players-tracked count P. The page refreshes on its own schedule, so a read
// shows rates as of the last refresh, and a refresh that adds many fresh
// players dilutes every rate down. We therefore model the two COUNT curves
// separately — earners E_i(t) and players P(t) — and take the rate as
// E_i(t) / P(t), so dilution falls out automatically when P grows faster
// than E_i. Integer earners are reconstructed from the rounded rates, the
// curves are fitted in log space against ln(1 + hours since launch), and
// the quantile spread is an honest random-walk + fit-residual + refresh
// uncertainty on the logit scale.
//
// Usage: MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=<dir> node main.ts
// Reads every *.in in the input dir (name order), writes <id>.out per case.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

// ---------------------------------------------------------------- constants
const Z = [-1.645, -0.674, 0, 0.674, 1.645]; // normal quantiles, levels .05/.25/.5/.75/.95
const EPOCH_MS = Date.parse("2026-11-19T00:00:00Z"); // launch anchor for log-time
const TIER_CAP: Record<string, number> = { platinum: 2, gold: 5, silver: 25, bronze: 90 };
const REFRESH_CV_FALLBACK = 0.18; // CV of P-jumps when none observed yet
const RESID_FALLBACK = 0.5; // logit residual when too few informative reads

// ------------------------------------------------------------------ helpers
function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function logit(p: number): number {
  const c = clamp(p, 0.0001, 0.9999);
  return Math.log(c / (1 - c));
}
function logistic(l: number): number {
  return 1 / (1 + Math.exp(-l));
}
function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}
function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / (xs.length - 1));
}
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n === 0 ? 0 : n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
// Least squares y = a + b*x. Returns null when x has no variance / <2 points.
function lstsq(xs: number[], ys: number[]): { a: number; b: number } | null {
  if (xs.length < 2) return null;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 0) return null;
  const b = sxy / sxx;
  return { a: my - b * mx, b };
}
function hoursSinceEpoch(iso: string): number {
  return Math.max(0, (Date.parse(iso) - EPOCH_MS) / 3600000);
}
function logTime(iso: string): number {
  return Math.log(1 + hoursSinceEpoch(iso));
}

// -------------------------------------------------------------------- types
type Trophy = { id: number; tier: string };
type Snap = {
  at: string;
  players: number;
  earned: number;
  rates: (number | null)[];
};
type Case = {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Snap[];
};

// ------------------------------------------------- dedupe + earner rebuild
// Drop consecutive identical reads (a frozen Exophase page): they carry no
// new information and would fake confidence in the fits.
function uniqueSnapshots(history: Snap[]): Snap[] {
  const out: Snap[] = [];
  for (const s of history) {
    const prev = out[out.length - 1];
    if (
      prev &&
      prev.players === s.players &&
      prev.earned === s.earned &&
      prev.rates.length === s.rates.length &&
      prev.rates.every((r, i) => r === s.rates[i])
    ) {
      continue;
    }
    out.push(s);
  }
  return out;
}

// A printed rate p (2 decimals) means the true rate lies in [p-0.005, p+0.005).
// With P players tracked, the integer earner count E lies in the interval
// below. p = 0 means E rounds to 0.00% (E below half a hundredth of a
// percent of P); p = 100 means every tracked player earned it, E = P.
function earnerInterval(pct: number, P: number): { lo: number; hi: number; center: number } {
  if (pct >= 100) return { lo: P, hi: P, center: P };
  if (pct <= 0) {
    const hi = Math.max(0, Math.ceil((0.005 / 100) * P) - 1);
    return { lo: 0, hi, center: 0 };
  }
  const lo = Math.max(0, Math.ceil(((pct - 0.005) / 100) * P));
  const hi = Math.max(0, Math.floor(((pct + 0.005) / 100) * P));
  return { lo, hi: Math.max(lo, hi), center: (pct / 100) * P };
}

// Reconstruct integer earners E[i][s] (trophy i, snapshot s) from the rounded
// rates: nearest integer to the printed rate inside its interval, forced
// monotone non-decreasing in time (earners never un-earn), then adjusted so
// the per-snapshot sum matches the page's trophies_earned count by
// largest-remainder moves inside each trophy's interval.
function reconstructEarners(snaps: Snap[], nTrophies: number): number[][] {
  const S = snaps.length;
  const E: number[][] = Array.from({ length: nTrophies }, () => new Array(S).fill(0));
  const LO: number[][] = Array.from({ length: nTrophies }, () => new Array(S).fill(0));
  const HI: number[][] = Array.from({ length: nTrophies }, () => new Array(S).fill(0));
  const CEN: number[][] = Array.from({ length: nTrophies }, () => new Array(S).fill(0));
  for (let s = 0; s < S; s++) {
    for (let i = 0; i < nTrophies; i++) {
      const r = snaps[s].rates[i];
      if (r === null || r === undefined) {
        // Missing rate in this read: carry the previous snapshot's count.
        const prev = s > 0 ? E[i][s - 1] : 0;
        E[i][s] = prev;
        LO[i][s] = prev;
        HI[i][s] = prev;
        CEN[i][s] = prev;
        continue;
      }
      const iv = earnerInterval(r, snaps[s].players);
      LO[i][s] = iv.lo;
      HI[i][s] = iv.hi;
      CEN[i][s] = iv.center;
      E[i][s] = clamp(Math.round(iv.center), iv.lo, iv.hi);
    }
  }
  // Monotone in time.
  for (let i = 0; i < nTrophies; i++) {
    for (let s = 1; s < S; s++) E[i][s] = Math.max(E[i][s], E[i][s - 1]);
  }
  // Match trophies_earned per snapshot (largest-remainder, interval-bounded,
  // never breaking monotonicity against the neighbouring snapshots).
  for (let s = 0; s < S; s++) {
    let guard = 0;
    for (;;) {
      const sum = E.reduce((a, row) => a + row[s], 0);
      const diff = snaps[s].earned - sum;
      if (diff === 0 || guard++ > 4 * nTrophies + 10) break;
      let best = -1;
      let bestScore = -Infinity;
      for (let i = 0; i < nTrophies; i++) {
        if (diff > 0) {
          const cap = Math.min(HI[i][s], s + 1 < S ? E[i][s + 1] : HI[i][s]);
          if (E[i][s] < cap && CEN[i][s] - E[i][s] > bestScore) {
            bestScore = CEN[i][s] - E[i][s];
            best = i;
          }
        } else {
          const floor = Math.max(LO[i][s], s > 0 ? E[i][s - 1] : LO[i][s]);
          if (E[i][s] > floor && E[i][s] - CEN[i][s] > bestScore) {
            bestScore = E[i][s] - CEN[i][s];
            best = i;
          }
        }
      }
      if (best < 0) break;
      E[best][s] += diff > 0 ? 1 : -1;
    }
  }
  return E;
}

// ----------------------------------------------------------------- forecast
type Row = { id: number; q: number[] };

function forecast(c: Case): Row[] {
  const n = c.trophies.length;
  const snaps = uniqueSnapshots(c.history);
  const last = c.history[c.history.length - 1];
  const lastRates: number[] = c.trophies.map((_, i) => {
    const r = last ? last.rates[i] : null;
    return r === null || r === undefined ? 0 : r;
  });
  const horizonDays = c.horizon_minutes / 1440;
  const xTarget = logTime(c.target);

  // Refresh-jump uncertainty: CV of the P jumps between distinct snapshots.
  const pJumps: number[] = [];
  for (let s = 1; s < snaps.length; s++) {
    const d = snaps[s].players - snaps[s - 1].players;
    if (d > 0) pJumps.push(d);
  }
  const cvDP =
    pJumps.length >= 2 && mean(pJumps) > 0
      ? std(pJumps) / mean(pJumps)
      : REFRESH_CV_FALLBACK;

  // ---- P(t): least squares log P vs ln(1+hours); clamp to [P_last, 4xP_last].
  const PLast = last ? last.players : 1;
  let PTarget = PLast;
  if (snaps.length >= 2) {
    const xs = snaps.map((s) => logTime(s.at));
    const ys = snaps.map((s) => Math.log(Math.max(1, s.players)));
    const fit = lstsq(xs, ys);
    if (fit) PTarget = clamp(Math.exp(fit.a + fit.b * xTarget), PLast, 4 * PLast);
  }

  // ---- E_i(t) per trophy, with tier-median slope shrinkage.
  const E = reconstructEarners(snaps, n);
  const xsAll = snaps.map((s) => logTime(s.at));
  // Raw per-trophy slopes first (for the tier medians).
  type Fit = { slope: number; resid: number; nInfo: number; eLast: number };
  const fits: (Fit | null)[] = [];
  for (let i = 0; i < n; i++) {
    const pts: { x: number; y: number }[] = [];
    for (let s = 0; s < snaps.length; s++) {
      if (snaps[s].rates[i] === null || snaps[s].rates[i] === undefined) continue;
      pts.push({ x: xsAll[s], y: Math.log(Math.max(0.5, E[i][s])) });
    }
    const distinctE = new Set(pts.map((p) => Math.round(Math.exp(p.y)))).size;
    const nInfo = Math.max(0, distinctE - 1);
    const fit = lstsq(
      pts.map((p) => p.x),
      pts.map((p) => p.y),
    );
    if (!fit || nInfo < 2) {
      fits.push({ slope: 0, resid: RESID_FALLBACK, nInfo, eLast: E[i][snaps.length - 1] });
      continue;
    }
    const slope = clamp(fit.b, 0, 3);
    let se = 0;
    for (const p of pts) se += (p.y - (fit.a + fit.b * p.x)) ** 2;
    const resid = pts.length > 0 ? Math.sqrt(se / pts.length) : RESID_FALLBACK;
    fits.push({ slope, resid, nInfo, eLast: E[i][snaps.length - 1] });
  }
  const tierMedianSlope: Record<string, number> = {};
  for (const tier of Object.keys(TIER_CAP)) {
    const slopes = c.trophies
      .map((t, i) => ({ t, f: fits[i] }))
      .filter((o) => o.t.tier === tier && o.f !== null && o.f.nInfo >= 2)
      .map((o) => (o.f as Fit).slope);
    tierMedianSlope[tier] = slopes.length > 0 ? median(slopes) : 0;
  }

  const fallback = snaps.length < 3;
  const rows: Row[] = [];
  for (let i = 0; i < n; i++) {
    const tier = c.trophies[i].tier;
    const cap = TIER_CAP[tier] ?? 90;
    const lastRate = lastRates[i];
    const f = fits[i] as Fit;

    let medRate: number; // median rate in percent
    let sigma: number;
    if (fallback) {
      // Fewer than 3 distinct snapshots (practice: the page never moved):
      // honest spread, median = last rate, no pretence of a trend.
      medRate = lastRate;
      sigma = 0.5 * Math.sqrt(horizonDays);
    } else {
      // Shrink this trophy's slope toward its tier median, weighted by how
      // many informative reads it has; refit the intercept at that slope.
      const w = f.nInfo / (f.nInfo + 2);
      const slopeS = w * f.slope + (1 - w) * (tierMedianSlope[tier] ?? 0);
      let aStar = 0;
      let cnt = 0;
      for (let s = 0; s < snaps.length; s++) {
        if (snaps[s].rates[i] === null || snaps[s].rates[i] === undefined) continue;
        aStar += Math.log(Math.max(0.5, E[i][s])) - slopeS * xsAll[s];
        cnt++;
      }
      aStar = cnt > 0 ? aStar / cnt : Math.log(Math.max(0.5, f.eLast));
      let eTarget = Math.exp(aStar + slopeS * xTarget);
      eTarget = Math.min(eTarget, PTarget);
      let rate = (eTarget / PTarget) * 100;
      rate = Math.min(rate, cap);
      medRate = rate;
      const resid = f.nInfo >= 3 ? f.resid : RESID_FALLBACK;
      sigma = clamp(Math.sqrt(0.25 * horizonDays + resid * resid), 0.15, 2.5);
      // Extra width for refresh timing we cannot see.
      sigma *= 1 + cvDP / 2;
    }
    // Rare trophies get a wider band (their counts are noisier in log space);
    // applies in both paths, exactly once.
    sigma *= Math.max(1, 0.005 / Math.max(medRate / 100, 1e-9));

    // ---- quantiles on the logit scale; a trophy whose last read is 100%
    // stays pinned at 100 (a 100% rate has no logit to spread around).
    let q: number[];
    if (lastRate >= 100) {
      q = [100, 100, 100, 100, 100];
    } else {
      const l = logit(medRate / 100);
      q = Z.map((z) => round4(clamp(100 * logistic(l + z * sigma), 0.01, 99.99)));
      for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    }
    rows.push({ id: c.trophies[i].id, q });
  }
  return rows;
}

// ---------------------------------------------------------------------- main
function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const raw = JSON.parse(readFileSync(join(inputDir, file), "utf-8"));
    // Normalise the case schema's field names to the internal ones:
    // history reads carry players_tracked / trophies_earned, not players / earned.
    const c: Case = {
      id: raw.id,
      at: raw.at,
      horizon_minutes: raw.horizon_minutes,
      target: raw.target,
      trophies: raw.trophies,
      history: (raw.history || []).map((h: any) => ({
        at: h.at,
        players: h.players_tracked,
        earned: h.trophies_earned,
        rates: h.rates,
      })),
    };
    const rows = forecast(c);
    // One line per trophy row; valid JSON, one object, exactly key "trophies".
    const body =
      '{"trophies": [\n' +
      rows.map((r) => JSON.stringify(r)).join(",\n") +
      "\n]}\n";
    writeFileSync(join(outputDir, c.id + ".out"), body, "utf-8");
  }
}

main();
