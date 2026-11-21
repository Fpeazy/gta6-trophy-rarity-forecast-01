// AdaptiveRefreshEnsemble_v2 — GTA VI trophy rarity forecast (rev4).
// Node 22, stdlib only, deterministic, no network.
// Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input/), writes one
// JSON output per case to $MAKEFUN_OUTPUT_DIR (else /output/).
// Method: frozen history -> predict the refresh itself (P x6.06, E x1.22);
// moving history -> counts fit + logit-trend fit ensemble (50/50 in logit
// space), shrunk 25% toward the last rate, then the moving-branch median
// is shifted DOWN 0.55 logit (late-refresh dilution prior).

import * as fs from "node:fs";
import * as path from "node:path";

const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const TIER_CAP: Record<string, number> = {
  platinum: 2,
  gold: 5,
  silver: 25,
  bronze: 90,
};

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function logit(pct: number): number {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}

function invLogit(l: number): number {
  const e = Math.exp(-clamp(l, -30, 30));
  return 100 / (1 + e);
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

// Least-squares fit of y against x; returns {intercept, slope}.
function linFit(xs: number[], ys: number[]): { a: number; b: number } {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  return { a: my - b * mx, b };
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function quantilesFromLogit(medianLogit: number, sigma: number): number[] {
  const qs = Z.map((z) => round4(clamp(invLogit(medianLogit + z * sigma), 0, 100)));
  for (let i = 1; i < qs.length; i++) {
    if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
}

interface TrophyDef {
  id: number;
  tier: string;
}

interface Snap {
  t: number; // hours since first read
  P: number;
  rates: number[];
}

function forecastCase(c: any): { trophies: { id: number; q: number[] }[] } {
  const trophies: TrophyDef[] = c.trophies.map((t: any) => ({
    id: t.id,
    tier: String(t.tier || "bronze").toLowerCase(),
  }));
  const n = trophies.length;
  const t0 = Date.parse(c.history[0].at);
  const targetMs = Date.parse(c.target);
  const atMs = Date.parse(c.at);
  const H_T = (targetMs - t0) / 3600000; // target hours since first read
  const D = (targetMs - atMs) / 86400000; // horizon in days

  // Step 2: dedup into unique snapshots (same P and identical rates = one).
  const snaps: Snap[] = [];
  for (const h of c.history) {
    const t = (Date.parse(h.at) - t0) / 3600000;
    const P = Number(h.players_tracked);
    const rates: number[] = (h.rates as number[]).map((r) => Number(r));
    const dup = snaps.some(
      (s) => s.P === P && s.rates.length === rates.length && s.rates.every((v, i) => v === rates[i]),
    );
    if (!dup) snaps.push({ t, P, rates });
  }
  snaps.sort((a, b) => a.t - b.t);
  const last = snaps[snaps.length - 1];

  // Step 3: integer earners per trophy per snapshot, monotone non-decreasing.
  const E: number[][] = []; // E[snap][trophy]
  for (const s of snaps) {
    E.push(
      s.rates.map((r, i) => {
        if (r >= 100) return s.P;
        return Math.round((r / 100) * s.P);
      }),
    );
  }
  for (let i = 0; i < n; i++) {
    for (let s = 1; s < snaps.length; s++) {
      if (E[s][i] < E[s - 1][i]) E[s][i] = E[s - 1][i];
    }
  }

  // Step 4: observed movement = max |Δlogit(rate)| over trophies, snapshots.
  let MOVE = 0;
  for (let i = 0; i < n; i++) {
    for (let s = 1; s < snaps.length; s++) {
      const d = Math.abs(logit(snaps[s].rates[i]) - logit(snaps[s - 1].rates[i]));
      if (d > MOVE) MOVE = d;
    }
  }

  const out: { id: number; q: number[] }[] = [];

  // Step 5: FROZEN branch (<=1 unique snapshot OR MOVE < 0.05).
  if (snaps.length <= 1 || MOVE < 0.05) {
    const P_t = last.P * 6.06;
    for (let i = 0; i < n; i++) {
      if (last.rates[i] >= 100) {
        out.push({ id: trophies[i].id, q: [100, 100, 100, 100, 100] });
        continue;
      }
      const E_t = E[snaps.length - 1][i] * 1.22;
      const rate = (E_t / P_t) * 100;
      out.push({ id: trophies[i].id, q: quantilesFromLogit(logit(rate), 0.35) });
    }
    return { trophies: out };
  }

  // Step 6: MOVING branch (>=2 unique snapshots AND MOVE >= 0.05).
  const xs = snaps.map((s) => Math.log(1 + s.t));
  const xT = Math.log(1 + H_T);

  // 6a: fit log P vs ln(1+hours), slope clamp [0,3].
  const fitP = linFit(xs, snaps.map((s) => Math.log(Math.max(1, s.P))));
  const slopeP = clamp(fitP.b, 0, 3);
  const P_final = Math.exp(fitP.a + slopeP * xT);

  // 6b: fit log E_i vs ln(1+hours), slope clamp [0,3], shrink 50% toward
  // the tier-median slope; E_i_final; rate_counts capped per tier.
  const tierOf = (i: number) => trophies[i].tier;
  const rawSlopes: number[] = [];
  const fitsE: { a: number; b: number }[] = [];
  for (let i = 0; i < n; i++) {
    const ys = E.map((row) => Math.log(Math.max(1, row[i])));
    const f = linFit(xs, ys);
    fitsE.push(f);
    rawSlopes.push(clamp(f.b, 0, 3));
  }
  const tierMedianSlope: Record<string, number> = {};
  for (const tier of Object.keys(TIER_CAP)) {
    const vals = rawSlopes.filter((_, i) => tierOf(i) === tier);
    tierMedianSlope[tier] = median(vals);
  }
  const rateCounts: number[] = [];
  const residAll: number[] = [];
  for (let i = 0; i < n; i++) {
    const bShrunk = 0.5 * rawSlopes[i] + 0.5 * (tierMedianSlope[tierOf(i)] ?? rawSlopes[i]);
    const E_final = Math.exp(fitsE[i].a + bShrunk * xT);
    let rc = (E_final / P_final) * 100;
    rc = clamp(rc, 0, TIER_CAP[tierOf(i)] ?? 90);
    rateCounts.push(rc);
    // RMS residual of the log-E fit (step 6f).
    const ys = E.map((row) => Math.log(Math.max(1, row[i])));
    let ss = 0;
    for (let s = 0; s < xs.length; s++) {
      const r = ys[s] - (fitsE[i].a + fitsE[i].b * xs[s]);
      ss += r * r;
    }
    residAll.push(Math.sqrt(ss / xs.length));
  }

  // 6c: fit logit(rate_i) vs ln(1+hours) -> rate_trend at H_T, tier-capped.
  const rateTrend: number[] = [];
  for (let i = 0; i < n; i++) {
    const ys = snaps.map((s) => logit(s.rates[i]));
    const f = linFit(xs, ys);
    let rt = invLogit(f.a + f.b * xT);
    rt = clamp(rt, 0, TIER_CAP[tierOf(i)] ?? 90);
    rateTrend.push(rt);
  }

  // 6f (shared quantities): P-jump CV and refresh-jump floor.
  const dP: number[] = [];
  for (let s = 1; s < snaps.length; s++) {
    const d = snaps[s].P - snaps[s - 1].P;
    if (d > 0) dP.push(d);
  }
  let cvDP = 0;
  if (dP.length >= 2) {
    const m = dP.reduce((a, b) => a + b, 0) / dP.length;
    const sd = Math.sqrt(dP.reduce((a, b) => a + (b - m) * (b - m), 0) / dP.length);
    cvDP = m > 0 ? sd / m : 0;
  }
  const jumps: number[] = [];
  for (let s = 1; s < snaps.length; s++) {
    if (snaps[s].P > snaps[s - 1].P * 1.05) {
      const ds: number[] = [];
      for (let i = 0; i < n; i++) {
        ds.push(Math.abs(logit(snaps[s].rates[i]) - logit(snaps[s - 1].rates[i])));
      }
      jumps.push(median(ds));
    }
  }
  const jumpFloor = jumps.length > 0 ? 0.5 * median(jumps) : 0;

  for (let i = 0; i < n; i++) {
    if (last.rates[i] >= 100) {
      out.push({ id: trophies[i].id, q: [100, 100, 100, 100, 100] });
      continue;
    }
    // 6d: median ensemble in logit space.
    const mLogit = 0.5 * logit(rateCounts[i]) + 0.5 * logit(rateTrend[i]);
    // 6e: persistence shrinkage, then the V8 change — shift median DOWN
    // 0.55 logit (late-refresh dilution prior).
    let mFinal = 0.75 * mLogit + 0.25 * logit(last.rates[i]);
    mFinal -= 0.55;
    const mRate = invLogit(mFinal);

    // 6f: spread.
    const resid = residAll[i];
    let sigmaBase = Math.sqrt(0.25 * D + resid * resid);
    sigmaBase = clamp(sigmaBase, 0.15, 2.5);
    const rateFrac = mRate / 100;
    const rareFloor = Math.max(1, 0.005 / (rateFrac + 1e-6));
    sigmaBase *= rareFloor * (1 + cvDP / 2);
    const sigma = Math.max(sigmaBase, jumpFloor, 0.3);

    // 6g: quantiles on logit(m_final) with sigma.
    out.push({ id: trophies[i].id, q: quantilesFromLogit(mFinal, sigma) });
  }
  return { trophies: out };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  fs.mkdirSync(outDir, { recursive: true });
  const files = fs
    .readdirSync(inDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const raw = fs.readFileSync(path.join(inDir, f), "utf8");
    const c = JSON.parse(raw);
    const result = forecastCase(c);
    const id = String(c.id);
    fs.writeFileSync(path.join(outDir, `${id}.out`), JSON.stringify(result));
  }
}

main();
