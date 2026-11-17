# GTA VI launch-day Twitch forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/uyekt1987/gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/uyekt1987/gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for make.fun `gta6-twitch-ccu-forecast-01`: forecast Grand Theft Auto VI's
launch-day (2026-11-19 UTC) Twitch concurrent viewers for hours 2..24, as five
quantiles per hour, conditioned on the hour-1 print.

## Model — v1-conditioned launch-shape forecaster

`main.ts` (`node main.ts`, Node 22, standard library only, offline, deterministic)
reads every `*.in` in `MAKEFUN_INPUT_DIR` (default `/input`) and writes one
`<id>.out` per case to `MAKEFUN_OUTPUT_DIR` (default `/output`).

1. **Anchor.** `p1` is the case's hour-1 print (floored at 1).
2. **Training launches.** From the case's `comparables`, keep only launches whose
   release-day hourly series is complete for hours 1..24, whose `unlock` text is
   tier-clean (no `early-access-contaminated`, no `Tier: weak`), and that unlocked
   at rolling midnight (`rolling midnight`, `midnight local` or `, rolling` in the
   unlock text). Rolling-midnight console launches are the right analogues for
   GTA VI; global-unlock games start near-empty in hour 1 and would poison the
   shape.
3. **Shape.** For each kept launch, `v1` is its hour-1 viewers and
   `r(h) = ln(v(h) / v1)` is the log growth of hour `h` over hour 1.
4. **Regression.** For each hour `h = 2..24`, ordinary least squares regresses
   `r(h)` on `ln(v1)` across the kept launches. The predicted median log-ratio at
   this case's `p1` is `m = a + b * ln(p1)`, clamped to `[-2, +3]`; the spread `s`
   is the residual standard deviation, floored at `0.25` log units.
5. **Quantiles.** `q_tau(h) = p1 * exp(m + z_tau * s)` with
   `z = [-1.645, -0.674, 0, 0.674, 1.645]`, rounded to 2 decimals, clamped to
   `(0, 1e9]`.
6. **Fallbacks.** Fewer than 5 kept launches → retry with all tier-clean
   comparables (kind filter dropped); still fewer than 5 → persistence baseline
   `q_tau(h) = p1 * exp(z_tau * 0.15 * sqrt(h - 1))`.

Output is validated against the contest contract (exactly 23 rows, hours 2..24,
five finite quantiles in `(0, 1e9]`, non-decreasing) before it is written.

## Local verification

On the 4 practice cases (scored against SullyGnome hourly truth):

| case | game | pinball loss |
| --- | --- | --- |
| p00 | Marvel's Wolverine | 0.19 |
| p01 | Red Dead Redemption 2 | 0.12 |
| p02 | God of War Ragnarök | 0.13 |
| p03 | Marvel's Spider-Man 2 | 0.23 |
| **mean** | | **≈ 0.165** |

House persistence baseline: 0.387. Runtime is ~50 ms per case, far under the
8 s cap.

## License

MIT — see `LICENSE`.
