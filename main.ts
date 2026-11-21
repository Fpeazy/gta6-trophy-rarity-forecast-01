// GTA VI trophy rarity forecast — AdaptiveRefreshEnsemble_v2 + ExactPosteriorFrozen (rev4 V9).
// Node 22, standard library only, no network, fully deterministic.
//
// Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input/) and writes one
// /output/<id>.out per case: {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}.
//
// Method in brief. Exophase refreshes its counts on its own schedule; a
// refresh that adds many fresh players dilutes every earn rate down. The
// program therefore works on the two count curves (earners E and players
// tracked P) and takes the rate as E/P, so dilution falls out when P grows
// faster than E. Which branch runs depends on whether the history moved:
//
//  FROZEN (one unique snapshot, or MOVE < 0.05): the exact posterior of the
//    refresh. Launch night's one refresh took P 638 -> 3,868 (x6.06) while
//    earners grew only ~x1.22, so the gain factor G is taken as
//    Uniform[1.08, 1.32] (centred on the inferred 1.22, wide enough to cover
//    the per-trophy gain spread plus 2-decimal rate quantisation noise):
//    P_new = P_last x 6.06, base_i = 100 x E_old,i / P_new, and
//    q_tau = base_i x (1.08 + 0.24 x tau) for tau in {0.05,0.25,0.5,0.75,0.95}.
//  MOVING (2+ unique snapshots and MOVE >= 0.05): an ensemble of a counts
//    fit (log E and log P against ln(1+hours), slopes clamped to [0,3] and
//    shrunk 50% toward the tier median, rate capped per tier) and a
//    logit-trend fit, averaged 50/50 in logit space, shrunk 25% toward the
//    last observed rate, then the median shifted DOWN 0.55 logit (late-
//    refresh dilution prior). Spread combines a horizon term, the log-E fit
//    residual, a rare-trophy floor, P-jump volatility, and a floor from the
//    observed refresh jumps, with a minimum sigma of 0.30.
// A trophy at 100% stays pinned at 100.

import * as fs from "node:fs";
import * as path from "node:path";

const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const TAUS = [0.05, 0.25, 0.5, 0.75, 0.95];
const TIER_CAP = { platinum: 2, gold: 5, silver: 25, bronze: 90 };
const REFRESH_P_GROWTH = 6.06; // launch night's one refresh: 638 -> 3,868 players
const GAIN_LO = 1.08; // exact-posterior frozen branch: G ~ Uniform[1.08, 1.32]
const GAIN_HI = 1.32;
const MOVING_MEDIAN_SHIFT = -0.55; // late-refresh dilution prior, logit units

function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}

function logitPct(rate) {
  const p = clamp(rate, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}

function invLogitPct(l) {
  return 100 / (1 + Math.exp(-l));
}

function round4(x) {
  return Math.round(x * 10000) / 10000;
}

// Quantiles from a logit median + sigma, clamped to [0,100], rounded to
// 4 decimals and forced non-decreasing.
function logitQuantiles(medianLogit, sigma) {
  const qs = Z.map((z) => round4(clamp(invLogitPct(medianLogit + z * sigma), 0, 100)));
  for (let i = 1; i < qs.length; i++) {
    if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return 0;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function mean(xs) {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}

// Least-squares straight line y = intercept + slope * x. Returns slope,
// intercept and the RMS residual of y.
function leastSquares(xs, ys) {
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx === 0 ? 0 : sxy / sxx;
  const intercept = my - slope * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    ss += r * r;
  }
  return { slope, intercept, rms: Math.sqrt(ss / Math.max(n, 1)) };
}

function forecastCase(kase) {
  const trophies = kase.trophies;
  const n = trophies.length;
  const firstAt = Date.parse(kase.history[0].at);
  const checkpointAt = Date.parse(kase.at);
  const targetAt = Date.parse(kase.target);
  const horizonDays = (targetAt - checkpointAt) / 86400000;
  const targetHours = (targetAt - firstAt) / 3600000;

  // --- Step 2: dedup history into unique snapshots (P + rates identical). ---
  const snaps = [];
  const seen = new Set();
  for (const h of kase.history) {
    const key = h.players_tracked + "|" + h.rates.map((r) => (r === null ? "" : r)).join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    snaps.push({
      hours: (Date.parse(h.at) - firstAt) / 3600000,
      p: h.players_tracked,
      rates: h.rates.slice(),
    });
  }
  const last = snaps[snaps.length - 1];

  // Fill a null rate by carrying the previous snapshot's rate forward
  // (0 when no earlier snapshot has one).
  for (let i = 0; i < n; i++) {
    let carry = 0;
    for (const s of snaps) {
      if (s.rates[i] === null || s.rates[i] === undefined) s.rates[i] = carry;
      else carry = s.rates[i];
    }
  }

  // --- Step 3: integer earners per trophy per snapshot, monotone. ---
  const earners = snaps.map((s) =>
    s.rates.map((r, i) => (r >= 100 ? s.p : Math.round((r / 100) * s.p)))
  );
  for (let i = 0; i < n; i++) {
    for (let k = 1; k < snaps.length; k++) {
      if (earners[k][i] < earners[k - 1][i]) earners[k][i] = earners[k - 1][i];
    }
  }

  // --- Step 4: observed movement, max |delta logit| over all trophies. ---
  let move = 0;
  for (let k = 1; k < snaps.length; k++) {
    for (let i = 0; i < n; i++) {
      const d = Math.abs(logitPct(snaps[k].rates[i]) - logitPct(snaps[k - 1].rates[i]));
      if (d > move) move = d;
    }
  }

  const out = [];
  const frozen = snaps.length <= 1 || move < 0.05;

  if (frozen) {
    // --- Step 5 (V9): FROZEN branch, exact posterior over the gain factor. ---
    const pNew = last.p * REFRESH_P_GROWTH;
    for (let i = 0; i < n; i++) {
      const lastRate = last.rates[i];
      if (lastRate >= 100) {
        out.push({ id: trophies[i].id, q: [100, 100, 100, 100, 100] });
        continue;
      }
      const eOld = earners[earners.length - 1][i];
      if (eOld === 0) {
        out.push({ id: trophies[i].id, q: [0, 0, 0, 0, 0] });
        continue;
      }
      const base = (100 * eOld) / pNew; // rate if the gain factor were 1
      const qs = TAUS.map((tau) =>
        round4(clamp(base * (GAIN_LO + (GAIN_HI - GAIN_LO) * tau), 0, 100))
      );
      for (let j = 1; j < qs.length; j++) {
        if (qs[j] < qs[j - 1]) qs[j] = qs[j - 1];
      }
      out.push({ id: trophies[i].id, q: qs });
    }
    return out;
  }

  // --- Step 6: MOVING branch (2+ unique snapshots and MOVE >= 0.05). ---
  const xs = snaps.map((s) => Math.log(1 + s.hours));
  const xTarget = Math.log(1 + targetHours);

  // 6a. Fit log P vs ln(1+hours); slope clamped to [0,3].
  const pFit = leastSquares(xs, snaps.map((s) => Math.log(s.p)));
  const pSlope = clamp(pFit.slope, 0, 3);
  const pFinal = Math.exp(pFit.intercept + pSlope * xTarget);

  // P-jump volatility: CV of the positive P increases between snapshots.
  const pIncreases = [];
  for (let k = 1; k < snaps.length; k++) {
    const d = snaps[k].p - snaps[k - 1].p;
    if (d > 0) pIncreases.push(d);
  }
  let cvP = 0;
  if (pIncreases.length >= 2) {
    const m = mean(pIncreases);
    const sd = Math.sqrt(mean(pIncreases.map((d) => (d - m) * (d - m))));
    cvP = m === 0 ? 0 : sd / m;
  }

  // Refresh-jump floor: for each P jump > 5% between consecutive snapshots,
  // the median |delta logit rate| over all trophies; floor is half the
  // median of those jumps (0 when there is none).
  const jumpMedians = [];
  for (let k = 1; k < snaps.length; k++) {
    if (snaps[k].p > snaps[k - 1].p * 1.05) {
      const deltas = [];
      for (let i = 0; i < n; i++) {
        deltas.push(Math.abs(logitPct(snaps[k].rates[i]) - logitPct(snaps[k - 1].rates[i])));
      }
      jumpMedians.push(median(deltas));
    }
  }
  const jumpFloor = jumpMedians.length === 0 ? 0 : 0.5 * median(jumpMedians);

  // Tier median slopes for the log-E fits (step 6b shrinkage target).
  const tierSlopes = {};
  for (const tier of ["platinum", "gold", "silver", "bronze"]) {
    const slopes = [];
    for (let i = 0; i < n; i++) {
      if (trophies[i].tier !== tier) continue;
      const ys = snaps.map((s, k) => Math.log(Math.max(earners[k][i], 0.5)));
      slopes.push(leastSquares(xs, ys).slope);
    }
    tierSlopes[tier] = slopes.length === 0 ? 0 : median(slopes);
  }

  for (let i = 0; i < n; i++) {
    const lastRate = last.rates[i];
    if (lastRate >= 100) {
      out.push({ id: trophies[i].id, q: [100, 100, 100, 100, 100] });
      continue;
    }
    const cap = TIER_CAP[trophies[i].tier] ?? 90;

    // 6b. Fit log E_i vs ln(1+hours); slope clamped [0,3], shrunk 50%
    // toward the tier median slope; rate from counts, capped per tier.
    const eYs = snaps.map((s, k) => Math.log(Math.max(earners[k][i], 0.5)));
    const eFit = leastSquares(xs, eYs);
    const eSlope = clamp(0.5 * clamp(eFit.slope, 0, 3) + 0.5 * tierSlopes[trophies[i].tier], 0, 3);
    const eFinal = Math.exp(eFit.intercept + eSlope * xTarget);
    const rateCounts = clamp((eFinal / pFinal) * 100, 0, cap);

    // 6c. Fit logit(rate_i) vs ln(1+hours); read at the target, capped.
    const rFit = leastSquares(xs, snaps.map((s) => logitPct(s.rates[i])));
    const rateTrend = clamp(invLogitPct(rFit.intercept + rFit.slope * xTarget), 0, cap);

    // 6d. Median ensemble in logit space, 50/50.
    let mLogit = 0.5 * logitPct(rateCounts) + 0.5 * logitPct(rateTrend);

    // 6e. Persistence shrinkage 25% toward the last rate, then the median
    // shifted DOWN 0.55 logit (late-refresh dilution prior).
    mLogit = 0.75 * mLogit + 0.25 * logitPct(lastRate);
    mLogit += MOVING_MEDIAN_SHIFT;

    // 6f. Spread: horizon term + log-E fit residual, rare-trophy floor,
    // P-jump volatility, refresh-jump floor, minimum sigma 0.30.
    let sigmaBase = Math.sqrt(0.25 * horizonDays + eFit.rms * eFit.rms);
    sigmaBase = clamp(sigmaBase, 0.15, 2.5);
    const mFinal = invLogitPct(mLogit);
    const rateFrac = mFinal / 100;
    const rareFloor = Math.max(1, 0.005 / (rateFrac + 1e-6));
    sigmaBase *= rareFloor * (1 + cvP / 2);
    const sigma = Math.max(sigmaBase, jumpFloor, 0.3);

    // 6g. Quantiles on logit(m_final) with sigma.
    out.push({ id: trophies[i].id, q: logitQuantiles(mLogit, sigma) });
  }
  return out;
}

function main() {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs.readdirSync(inputDir).filter((f) => f.endsWith(".in")).sort();
  for (const file of files) {
    const kase = JSON.parse(fs.readFileSync(path.join(inputDir, file), "utf8"));
    const rows = forecastCase(kase);
    const text = JSON.stringify({ trophies: rows });
    fs.writeFileSync(path.join(outputDir, kase.id + ".out"), text, "utf8");
  }
}

main();
