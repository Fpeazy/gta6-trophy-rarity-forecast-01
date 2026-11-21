// AdaptiveRefreshEnsemble_v1 — GTA VI trophy rarity forecast, rev3.
// Forecasts each trophy's Exophase earn rate at the case target as five
// quantiles (5/25/50/75/95%). Node 22, standard library only, no network,
// fully deterministic.
//
// Idea: Exophase refreshes its counts on its own schedule. A refresh that
// adds many fresh players dilutes every rate down, so the model works in
// counts (earners E and players tracked P) and takes the rate as E/P.
//   * FROZEN branch (history never moved): the practice cases show one
//     refresh multiplies P by ~6.06 while earners grow only ~1.22x, so the
//     forecast rate is last_rate x 1.22/6.06 with a wide logit spread.
//   * MOVING branch (history moved): ensemble of a counts fit (log E and
//     log P against ln(1+hours), tier-shrunk slopes) and a logit-trend fit,
//     averaged 50/50 in logit space, shrunk 25% toward the last rate, with
//     a spread that grows with horizon, fit residual, rarity, P-jump
//     volatility and the observed refresh-jump size.
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const TIER_CAP: Record<string, number> = {
  platinum: 2,
  gold: 5,
  silver: 25,
  bronze: 90,
};
// Observed on launch night: one refresh took P 638 -> 3868 (x6.06) while
// earners grew only ~1.22x. These are the frozen-branch refresh factors.
const REFRESH_P = 6.06;
const REFRESH_E = 1.22;
const FROZEN_SIGMA = 0.35;

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function logitPct(rate: number): number {
  const p = clamp(rate, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}

function invLogitPct(l: number): number {
  const p = 1 / (1 + Math.exp(-l));
  return clamp(p * 100, 0, 100);
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function quantilesFromLogit(centerLogit: number, sigma: number): number[] {
  const qs = Z.map((z) => round4(invLogitPct(centerLogit + z * sigma)));
  for (let i = 1; i < qs.length; i++) {
    if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

// Least squares y = a + b*x, plus the RMS residual of the fit.
function lsq(xs: number[], ys: number[]): { a: number; b: number; resid: number } {
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    ss += r * r;
  }
  return { a, b, resid: Math.sqrt(ss / n) };
}

interface Snap {
  h: number; // hours since the first read
  P: number; // players tracked
  rates: number[]; // earn rate per trophy, percent
}

function forecastCase(c: any): { trophies: { id: number; q: number[] }[] } {
  const trophies: any[] = c.trophies;
  const n = trophies.length;
  const firstAt = Date.parse(c.history[0].at);
  const atMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const hTarget = (targetMs - firstAt) / 3600000; // target, hours since first read
  const horizonDays = (targetMs - atMs) / 60000 / 1440; // horizon in days

  // 1. Deduplicate history into unique snapshots (same P and same rates = one).
  const snaps: Snap[] = [];
  const seen = new Set<string>();
  // A read can lack a trophy (null rate); carry its last seen rate forward
  // instead of treating the trophy as never earned.
  const carried = new Array<number>(n).fill(0);
  for (const r of c.history) {
    const P = r.players_tracked as number;
    const rates = (r.rates as (number | null)[]).map((v, i) => {
      if (v === null || v === undefined) return carried[i];
      carried[i] = v;
      return v;
    });
    const key = P + "|" + rates.join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    snaps.push({ h: (Date.parse(r.at) - firstAt) / 3600000, P, rates });
  }

  const last = snaps[snaps.length - 1];
  const lastRates = last.rates;

  // 2. Reconstruct integer earners E per trophy per snapshot, monotone
  // non-decreasing over time; a trophy at 100% has E = P.
  const E: number[][] = []; // E[s][i]
  const prevE = new Array<number>(n).fill(0);
  for (const s of snaps) {
    const row: number[] = [];
    for (let i = 0; i < n; i++) {
      let e: number;
      if (s.rates[i] >= 100) {
        e = s.P;
      } else {
        e = Math.round((s.rates[i] / 100) * s.P);
      }
      if (e < prevE[i]) e = prevE[i];
      prevE[i] = e;
      row.push(e);
    }
    E.push(row);
  }

  // 3. Observed movement: largest |delta logit(rate)| over any trophy,
  // between consecutive unique snapshots.
  let move = 0;
  for (let s = 1; s < snaps.length; s++) {
    for (let i = 0; i < n; i++) {
      const d = Math.abs(logitPct(snaps[s].rates[i]) - logitPct(snaps[s - 1].rates[i]));
      if (d > move) move = d;
    }
  }

  const out: { id: number; q: number[] }[] = [];

  // 4. FROZEN branch: at most one unique snapshot, or nothing ever moved.
  if (snaps.length <= 1 || move < 0.05) {
    const pTarget = last.P * REFRESH_P;
    for (let i = 0; i < n; i++) {
      if (lastRates[i] >= 100) {
        out.push({ id: trophies[i].id, q: [100, 100, 100, 100, 100] });
        continue;
      }
      const eTarget = E[E.length - 1][i] * REFRESH_E;
      const rate = (eTarget / pTarget) * 100;
      out.push({ id: trophies[i].id, q: quantilesFromLogit(logitPct(rate), FROZEN_SIGMA) });
    }
    return { trophies: out };
  }

  // 5. MOVING branch.
  const xs = snaps.map((s) => Math.log(1 + s.h));
  const xTarget = Math.log(1 + hTarget);

  // 5a. Fit log P vs ln(1+hours); slope clamped to [0, 3].
  const pFit = lsq(xs, snaps.map((s) => Math.log(Math.max(s.P, 1))));
  const pSlope = clamp(pFit.b, 0, 3);
  const pFinal = Math.exp(pFit.a + pSlope * xTarget);

  // 5b. Fit log E_i vs ln(1+hours) per trophy; slope clamped to [0, 3],
  // then shrunk 50% toward its tier's median slope.
  const rawSlopes: number[] = [];
  const eFits: { a: number; b: number; resid: number }[] = [];
  for (let i = 0; i < n; i++) {
    const ys = snaps.map((s, si) => Math.log(Math.max(E[si][i], 0.5)));
    const f = lsq(xs, ys);
    eFits.push(f);
    rawSlopes.push(clamp(f.b, 0, 3));
  }
  const tierMedianSlope: Record<string, number> = {};
  for (const tier of Object.keys(TIER_CAP)) {
    const slopes = trophies
      .map((t, i) => (t.tier === tier ? rawSlopes[i] : null))
      .filter((v): v is number => v !== null);
    tierMedianSlope[tier] = median(slopes);
  }
  const rateCounts: number[] = [];
  const resids: number[] = [];
  for (let i = 0; i < n; i++) {
    const med = tierMedianSlope[trophies[i].tier] ?? rawSlopes[i];
    const slope = 0.5 * rawSlopes[i] + 0.5 * med;
    const eFinal = Math.exp(eFits[i].a + slope * xTarget);
    const cap = TIER_CAP[trophies[i].tier] ?? 90;
    rateCounts.push(clamp((eFinal / pFinal) * 100, 0, cap));
    resids.push(eFits[i].resid);
  }

  // 5c. Fit logit(rate_i) vs ln(1+hours) per trophy, read at the target.
  const rateTrend: number[] = [];
  for (let i = 0; i < n; i++) {
    const ys = snaps.map((s) => logitPct(s.rates[i]));
    const f = lsq(xs, ys);
    const cap = TIER_CAP[trophies[i].tier] ?? 90;
    rateTrend.push(clamp(invLogitPct(f.a + f.b * xTarget), 0, cap));
  }

  // P jumps between consecutive snapshots, for the spread terms.
  const pJumps: number[] = [];
  const jumpMedians: number[] = [];
  for (let s = 1; s < snaps.length; s++) {
    const dP = snaps[s].P - snaps[s - 1].P;
    if (dP > 0) pJumps.push(dP);
    if (snaps[s - 1].P > 0 && dP / snaps[s - 1].P > 0.05) {
      const deltas: number[] = [];
      for (let i = 0; i < n; i++) {
        deltas.push(Math.abs(logitPct(snaps[s].rates[i]) - logitPct(snaps[s - 1].rates[i])));
      }
      jumpMedians.push(median(deltas));
    }
  }
  let cvP = 0;
  if (pJumps.length >= 2) {
    const m = mean(pJumps);
    if (m > 0) {
      const varP = mean(pJumps.map((v) => (v - m) * (v - m)));
      cvP = Math.sqrt(varP) / m;
    }
  }
  const jumpFloor = jumpMedians.length > 0 ? 0.5 * median(jumpMedians) : 0;

  for (let i = 0; i < n; i++) {
    if (lastRates[i] >= 100) {
      out.push({ id: trophies[i].id, q: [100, 100, 100, 100, 100] });
      continue;
    }
    // 5d. Ensemble median in logit space: counts fit + trend fit, 50/50.
    const logitM =
      0.5 * logitPct(rateCounts[i]) + 0.5 * logitPct(rateTrend[i]);
    // 5e. Persistence shrinkage: 75% ensemble, 25% last observed rate.
    const logitFinal = 0.75 * logitM + 0.25 * logitPct(lastRates[i]);
    const mFinal = invLogitPct(logitFinal);
    // 5f. Spread: horizon + fit residual, rare-trophy floor, P-jump
    // volatility, then a floor from the observed refresh-jump size.
    let sigmaBase = Math.sqrt(0.25 * horizonDays + resids[i] * resids[i]);
    sigmaBase = clamp(sigmaBase, 0.15, 2.5);
    const rateFrac = mFinal / 100;
    const rareFloor = Math.max(1, 0.005 / (rateFrac + 1e-6));
    sigmaBase = sigmaBase * rareFloor * (1 + cvP / 2);
    const sigma = Math.max(sigmaBase, jumpFloor, 0.3);
    // 5g. Quantiles on the logit scale.
    out.push({ id: trophies[i].id, q: quantilesFromLogit(logitFinal, sigma) });
  }
  return { trophies: out };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const c = JSON.parse(readFileSync(join(inputDir, f), "utf8"));
    const out = forecastCase(c);
    writeFileSync(join(outputDir, f.replace(/\.in$/, ".out")), JSON.stringify(out));
  }
}

main();
