// gta6-twitch-ccu-forecast-01 — v1-conditioned launch-shape forecaster.
//
// Forecast: for each hour h = 2..24 of release day, five quantiles of the Twitch
// concurrent-viewer count in the launch category, conditioned on the hour-1 print.
//
// Model (per case):
//   * p1 is the hour-1 print (floored at 1).
//   * Training launches come from the case's `comparables`. A comparable is kept
//     only when its release-day hourly series is complete for hours 1..24, its
//     `unlock` text is tier-clean (no "early-access-contaminated", no "Tier: weak"),
//     and it is a rolling-midnight launch (unlock mentions "rolling midnight",
//     "midnight local" or ", rolling"). Rolling-midnight console launches are the
//     right analogues for GTA VI; global-unlock games start near-empty in hour 1
//     and would poison the shape.
//   * For each kept launch, v1 is its hour-1 viewers and r(h) = ln(v(h) / v1) is
//     the log growth of hour h over hour 1.
//   * For each hour h, ordinary least squares regresses r(h) on ln(v1) across the
//     kept launches. The predicted median log-ratio at this case's p1 is
//     m = a + b * ln(p1), clamped to [-2, +3]. The spread s is the residual
//     standard deviation of that regression, floored at 0.25 log units.
//   * Quantiles: q_tau(h) = p1 * exp(m + z_tau * s), z = [-1.645, -0.674, 0,
//     0.674, 1.645], rounded to 2 decimals and clamped to (0, 1e9].
//   * Fallbacks: fewer than 5 kept launches -> retry with all tier-clean
//     comparables (kind filter dropped); still fewer than 5 -> persistence
//     baseline q_tau(h) = p1 * exp(z_tau * 0.15 * sqrt(h - 1)).
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

/** Population standard deviation. */
const stddev = (xs: number[]): number => {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) * (x - m))));
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

  const logV1 = group.map((g) => Math.log(g.v1));
  const meanLogV1 = mean(logV1);

  return HOURS.map((h, hi) => {
    const ys = group.map((g) => g.logRatio[hi]);
    const meanY = mean(ys);
    // Ordinary least squares of r(h) on ln(v1).
    let sxy = 0;
    let sxx = 0;
    for (let i = 0; i < logV1.length; i++) {
      sxy += (logV1[i] - meanLogV1) * (ys[i] - meanY);
      sxx += (logV1[i] - meanLogV1) * (logV1[i] - meanLogV1);
    }
    const slope = sxx > 0 ? sxy / sxx : 0;
    const intercept = meanY - slope * meanLogV1;
    const residuals = ys.map((y, i) => y - (intercept + slope * logV1[i]));
    const spread = Math.max(SPREAD_FLOOR, stddev(residuals));
    const medianLogRatio = Math.max(-MEDIAN_CLAMP, Math.min(MEDIAN_CLAMP, intercept + slope * logP1));
    return { hour: h, q: quantilesFrom(p1, medianLogRatio, spread) };
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
