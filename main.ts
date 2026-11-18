// gta6-twitch-ccu-forecast-01 — v2: three-way blended launch-shape forecaster.
//
// Forecast: for each hour h = 2..24 of release day, five quantiles of the Twitch
// concurrent-viewer count in the launch category, conditioned on the hour-1 print.
//
// Why v2 (replaces the v1 single OLS + Gaussian-spread model, practice mean
// pinball 0.1637)? Diagnostics on the 11 tier-clean rolling-midnight comparables
// showed two systematic flaws in v1:
//   * Residuals of log-ratio r(h) = ln(v(h)/v1) are strongly positively skewed
//     (skew +0.8..+1.6 for h >= 6; 10th percentile ~ -0.45, 90th ~ +0.25 around
//     the fitted median). A symmetric Gaussian spread therefore puts the 5%
//     quantile too low and the 95% quantile too high.
//   * Direct pinball minimisation (quantile regression) beats the Gaussian
//     approximation on held-out launches: leave-one-out pinball over the 11
//     rolling comparables 0.235 vs 0.255 for v1. But quantile regression is
//     unstable when the target hour-1 print extrapolates outside the training
//     ln(v1) range (practice case p00, below range, got worse), so it is used
//     only inside the training range.
//
// Model (per case):
//   * p1 is the hour-1 print (floored at 1); lnP1 = ln(p1).
//   * Training launches come from the case's `comparables`, same selection as
//     v1: complete release-day series for hours 1..24, tier-clean (no
//     "early-access-contaminated", no "Tier: weak"), rolling-midnight unlock
//     (unlock mentions "rolling midnight", "midnight local" or ", rolling").
//     Fewer than 5 kept launches -> retry with all tier-clean comparables
//     (kind filter dropped); still fewer than 5 -> persistence baseline.
//   * Three per-hour component forecasts are computed from the kept group and
//     blended in log space with weights 0.4 / 0.4 / 0.2:
//       1. OLS: per-hour OLS of r(h) on ln(v1); median m = a + b*lnP1 clamped
//          to [-3, +3]; spread s = residual std floored at 0.25; quantiles
//          p1*exp(m + z*s), z = [-1.645, -0.674, 0, 0.674, 1.645]. (This is
//          the v1 model, kept as the stable anchor.)
//       2. Quantile regression: per hour and per level tau, the (a, b) pair
//          minimising the pinball loss of r(h) on ln(v1), found by grid search
//          over slope b in [-1.50, 0.60] (step 0.01) with the intercept
//          re-solved exactly as the tau-quantile of the residuals. The slope
//          is then clamped to [-0.60, +0.10] (the range the data supports;
//          wider slopes are noise from n ~ 11) and the intercept re-solved
//          for the clamped slope. Quantiles p1*exp(a + b*lnP1).
//          USED ONLY when lnP1 lies inside the training ln(v1) range; outside
//          it the weight moves to the other two components (renormalised to
//          2/3 OLS + 1/3 empirical), because quantile slopes extrapolate badly.
//       3. Empirical residuals: the OLS median from (1) plus the empirical
//          quantiles of that hour's OLS residuals at the five levels,
//          smoothed across neighbouring hours with a 3-hour moving average
//          (per-hour empirical quantiles from ~11 residuals are noisy; the
//          residual distribution changes slowly with h). This captures the
//          positive skew the Gaussian misses.
//   * Blend: q_tau(h) = exp(0.4*ln qOLS + 0.4*ln qQR + 0.2*ln qEmp), with the
//     trust-region renormalisation above; rows are then forced non-decreasing
//     and clamped to (0, 1e9].
//   * Fallbacks: same as v1 (persistence baseline q_tau(h) =
//     p1*exp(z_tau * 0.15 * sqrt(h - 1)) when the group is too small).
//
// Validation (local scorer, 4 practice cases; LOO = leave-one-out pinball on
// comparables, training group = rolling-midnight clean launches):
//   * Practice mean pinball: v2 0.1586 vs v1 0.1637. Per case: p00 0.1848 (was
//     0.1862), p01 0.1164 (was 0.1164), p02 0.1151 (was 0.1286), p03 0.2180
//     (was 0.2237). No case worsens.
//   * LOO rolling (n=11): v2 0.2427 vs v1 0.2551. LOO clean-all, same
//     rolling-only training (n=26 targets): v2 0.6706 vs v1 0.6709.
//   * Rejected along the way (practice win but LOO loss, or vice versa):
//     pooled/shrunk slopes, Huber, Theil-Sen, similarity weighting, shape
//     ensembles, history/pre-launch features, EDT dummy, kNN shapes.
//   * Caveat: 4 practice cases and 11 LOO launches are thin evidence, and the
//     final case is a single unseen launch; the ~3-5% gains above are real on
//     the available data but do not guarantee a better final score.
//
// Deterministic, standard library only, no network, no randomness.
// Usage: MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=<dir> node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Scored quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard-normal quantiles at LEVELS. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Hours of release day to forecast. */
const HOURS = Array.from({ length: 23 }, (_, i) => i + 2);
/** Minimum kept launches for the shape model; below this the fallbacks apply. */
const MIN_GROUP = 5;
/** Median log-ratio is clamped to this range so one extreme case cannot explode it. */
const MEDIAN_CLAMP = 3;
/** Residual spread is floored at this many log units. */
const SPREAD_FLOOR = 0.25;
/** Persistence baseline spread, log units per square-root hour. */
const BASELINE_SIGMA = 0.15;
/** Largest allowed quantile (the contest cap). */
const MAX_VIEWERS = 1e9;
/** Blend weights for the three components: OLS, quantile regression, empirical. */
const W_OLS = 0.4;
const W_QR = 0.4;
const W_EMP = 0.2;
/** Quantile-regression slope search grid. */
const QR_SLOPE_MIN = -1.5;
const QR_SLOPE_MAX = 0.6;
const QR_SLOPE_STEP = 0.01;
/** Quantile-regression slopes are clamped to this range (the data supports no more). */
const QR_SLOPE_CLAMP = 0.6;
const QR_SLOPE_CLAMP_HI = 0.1;

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** One kept training launch: its hour-1 viewers and its log growth per hour. */
interface ShapeLaunch {
  v1: number;
  logRatio: number[]; // index h - 2 for h = 2..24
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

/** Population standard deviation. */
const stddev = (xs: number[]): number => {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) * (x - m))));
};

/** Linear-interpolation quantile of a sample at probability p. */
const sampleQuantile = (xs: number[], p: number): number => {
  const s = [...xs].sort((a, b) => a - b);
  const idx = (s.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
};

/**
 * Release-day hourly map of one comparable: hour h (1..24) -> viewers, from the
 * point whose `from` is day_start + (h - 1) hours, keeping only positive counts.
 */
const releaseDayMap = (comp: Comparable): Map<number, number> => {
  const start = Date.parse(comp.day_start);
  const map = new Map<number, number>();
  for (const point of comp.hourly ?? []) {
    const h = Math.round((Date.parse(point.from) - start) / 3_600_000) + 1;
    if (h >= 1 && h <= 24 && point.viewers > 0) map.set(h, point.viewers);
  }
  return map;
};

/** True when the comparable's unlock text marks it early-access-contaminated or weak. */
const isTierExcluded = (comp: Comparable): boolean => {
  const text = comp.unlock ?? '';
  return text.includes('early-access-contaminated') || text.includes('Tier: weak');
};

/** True when the comparable unlocked at midnight local time, rolling by region. */
const isRollingMidnight = (comp: Comparable): boolean => {
  const text = (comp.unlock ?? '').toLowerCase();
  return text.includes('rolling midnight') || text.includes('midnight local') || text.includes(', rolling');
};

/**
 * Build the training group: tier-clean comparables with a complete 1..24 release
 * day; when `rollingOnly` is set, keep rolling-midnight launches only.
 */
const trainingGroup = (comparables: Comparable[], rollingOnly: boolean): ShapeLaunch[] => {
  const group: ShapeLaunch[] = [];
  for (const comp of comparables ?? []) {
    if (isTierExcluded(comp)) continue;
    if (rollingOnly && !isRollingMidnight(comp)) continue;
    const day = releaseDayMap(comp);
    let complete = true;
    for (let h = 1; h <= 24; h++) {
      if (!day.has(h)) {
        complete = false;
        break;
      }
    }
    if (!complete) continue;
    const v1 = day.get(1) as number;
    const logRatio = HOURS.map((h) => Math.log(day.get(h) as number / v1));
    group.push({ v1, logRatio });
  }
  return group;
};

/** Clamp a quantile into (0, 1e9] after rounding to 2 decimals. */
const clampQuantile = (x: number): number => {
  const r = round2(x);
  if (!Number.isFinite(r) || r <= 0) return 0.01;
  return Math.min(MAX_VIEWERS, r);
};

/**
 * The five quantiles for one hour from a median log-ratio m and spread s.
 * Non-decreasing by construction (z increases, s > 0); enforced anyway after
 * rounding, so the output can never violate the contract.
 */
const quantilesFrom = (p1: number, m: number, s: number): number[] => {
  const qs = Z.map((z) => clampQuantile(p1 * Math.exp(m + z * s)));
  for (let i = 1; i < qs.length; i++) {
    if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
};

/** Persistence baseline quantiles for one hour (the house baseline's formula). */
const baselineQuantiles = (p1: number, h: number): number[] =>
  quantilesFrom(p1, 0, BASELINE_SIGMA * Math.sqrt(h - 1));

/** Ordinary least squares of ys on xs. */
const ols = (xs: number[], ys: number[]): { a: number; b: number; resid: number[] } => {
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  const a = my - b * mx;
  return { a, b, resid: ys.map((y, i) => y - (a + b * xs[i])) };
};

/**
 * Pinball loss of intercept `a` at level tau for residuals `res` (already
 * net of the slope term), used to re-solve the intercept exactly once a slope
 * is fixed: the minimiser is a sample quantile, so checking every residual
 * as a candidate is exact for n this small.
 */
const bestIntercept = (res: number[], tau: number): number => {
  const srt = [...res].sort((a, b) => a - b);
  let bestA = srt[0];
  let bestLoss = Infinity;
  for (const cand of srt) {
    let loss = 0;
    for (const r of res) {
      const d = r - cand;
      loss += d >= 0 ? tau * d : (tau - 1) * d;
    }
    if (loss < bestLoss) {
      bestLoss = loss;
      bestA = cand;
    }
  }
  return bestA;
};

/**
 * Direct quantile regression for one hour and level: grid-search the slope
 * minimising pinball loss, then clamp the slope to the supported range and
 * re-solve the intercept for the clamped slope.
 */
const quantRegLine = (xs: number[], ys: number[], tau: number): { a: number; b: number } => {
  let bestA = 0;
  let bestB = 0;
  let bestLoss = Infinity;
  for (let bi = 0; bi * QR_SLOPE_STEP <= QR_SLOPE_MAX - QR_SLOPE_MIN + 1e-9; bi++) {
    const b = QR_SLOPE_MIN + bi * QR_SLOPE_STEP;
    const res = ys.map((y, i) => y - b * xs[i]);
    const a = bestIntercept(res, tau);
    let loss = 0;
    for (const r of res) {
      const d = r - a;
      loss += d >= 0 ? tau * d : (tau - 1) * d;
    }
    if (loss < bestLoss) {
      bestLoss = loss;
      bestA = a;
      bestB = b;
    }
  }
  const bClamped = Math.max(-QR_SLOPE_CLAMP, Math.min(QR_SLOPE_CLAMP_HI, bestB));
  if (bClamped === bestB) return { a: bestA, b: bestB };
  const res = ys.map((y, i) => y - bClamped * xs[i]);
  return { a: bestIntercept(res, tau), b: bClamped };
};

/**
 * The three component quantile sets for one case, each as [hour][level] log
 * values (log of the viewer count, so blending is a weighted average).
 * Component 1 (OLS + Gaussian) and component 3 (OLS median + smoothed
 * empirical residual quantiles) share the OLS fit; component 2 is direct
 * quantile regression. Returns null when the group is too small.
 */
const componentLogQuantiles = (
  group: ShapeLaunch[],
  p1: number,
  logP1: number,
): { ols: number[][]; qr: number[][]; emp: number[][] } | null => {
  if (group.length < MIN_GROUP) return null;
  const logV1 = group.map((g) => Math.log(g.v1));
  const olsLog: number[][] = [];
  const qrLog: number[][] = [];
  const empLog: number[][] = [];
  // Per-hour OLS medians and empirical residual-quantile offsets, kept to
  // smooth the offsets across neighbouring hours afterwards.
  const medians: number[] = [];
  const offsets: number[][] = [];
  for (let hi = 0; hi < HOURS.length; hi++) {
    const ys = group.map((g) => g.logRatio[hi]);
    const fit = ols(logV1, ys);
    const spread = Math.max(SPREAD_FLOOR, stddev(fit.resid));
    const m = Math.max(-MEDIAN_CLAMP, Math.min(MEDIAN_CLAMP, fit.a + fit.b * logP1));
    medians.push(m);
    olsLog.push(Z.map((z) => Math.log(p1) + m + z * spread));
    qrLog.push(
      LEVELS.map((tau) => {
        const line = quantRegLine(logV1, ys, tau);
        return Math.log(p1) + line.a + line.b * logP1;
      }),
    );
    offsets.push(LEVELS.map((tau) => sampleQuantile(fit.resid, tau)));
  }
  for (let hi = 0; hi < HOURS.length; hi++) {
    const lo = Math.max(0, hi - 1);
    const hi2 = Math.min(HOURS.length - 1, hi + 1);
    const row: number[] = [];
    for (let j = 0; j < LEVELS.length; j++) {
      let s = 0;
      let n = 0;
      for (let k = lo; k <= hi2; k++) {
        s += offsets[k][j];
        n++;
      }
      row.push(Math.log(p1) + medians[hi] + s / n);
    }
    empLog.push(row);
  }
  return { ols: olsLog, qr: qrLog, emp: empLog };
};

/**
 * Forecast one case. Returns the 23 output rows for hours 2..24.
 */
const forecastCase = (c: CaseFile): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  // A non-finite print cannot anchor a forecast; fall back to 1 viewer.
  const p1 = Number.isFinite(first.viewers) ? Math.max(1, first.viewers) : 1;
  const logP1 = Math.log(p1);

  let group = trainingGroup(c.comparables, true);
  if (group.length < MIN_GROUP) group = trainingGroup(c.comparables, false);
  if (group.length < MIN_GROUP) {
    return HOURS.map((h) => ({ hour: h, q: baselineQuantiles(p1, h) }));
  }

  const comps = componentLogQuantiles(group, p1, logP1);
  if (comps === null) {
    return HOURS.map((h) => ({ hour: h, q: baselineQuantiles(p1, h) }));
  }

  // Trust region: quantile-regression slopes are only trusted while the
  // target hour-1 print sits inside the training ln(v1) range. Outside it
  // the QR weight is dropped and the other two renormalised.
  const logV1 = group.map((g) => Math.log(g.v1));
  const inRange = logP1 >= Math.min(...logV1) && logP1 <= Math.max(...logV1);
  let wOls = W_OLS;
  let wQr = W_QR;
  let wEmp = W_EMP;
  if (!inRange) {
    const s = wOls + wEmp;
    wOls /= s;
    wEmp /= s;
    wQr = 0;
  }

  return HOURS.map((h, hi) => {
    const qs = LEVELS.map((_, j) => {
      const lq = wOls * comps.ols[hi][j] + wQr * comps.qr[hi][j] + wEmp * comps.emp[hi][j];
      return clampQuantile(Math.exp(lq));
    });
    for (let i = 1; i < qs.length; i++) {
      if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
    }
    return { hour: h, q: qs };
  });
};

/** Validate one case's output rows against the contest contract before writing. */
const validateRows = (rows: Array<{ hour: number; q: number[] }>): void => {
  if (rows.length !== 23) throw new Error(`expected 23 rows, got ${rows.length}`);
  const seen = new Set<number>();
  for (const row of rows) {
    if (row.hour < 2 || row.hour > 24 || seen.has(row.hour)) {
      throw new Error(`bad or duplicate hour ${row.hour}`);
    }
    seen.add(row.hour);
    if (row.q.length !== LEVELS.length) throw new Error(`hour ${row.hour}: expected 5 quantiles`);
    for (let i = 0; i < row.q.length; i++) {
      const q = row.q[i];
      if (!Number.isFinite(q) || q <= 0 || q > MAX_VIEWERS) {
        throw new Error(`hour ${row.hour}: quantile ${q} out of range`);
      }
      if (i > 0 && q < row.q[i - 1]) throw new Error(`hour ${row.hour}: quantiles not non-decreasing`);
    }
  }
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const hours = forecastCase(c);
  validateRows(hours);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
