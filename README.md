# GTA VI Trophy Rarity Forecast

[![Bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fpeazy/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fpeazy/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts every PlayStation 5 trophy's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
for Grand Theft Auto VI at 2026-11-26T00:00:00Z — a [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty entry
(play code `afc5cnfr`).

## Method

Counts model, written from scratch in TypeScript (Node 22, standard library
only, no network, fully deterministic):

- From each history read, recover earners `E = rate × players / 100` and
  players tracked `P`.
- Fit `log E` and `log P` separately against `ln(1 + hours since
  2026-11-19T00:00:00Z)` by least squares, so the forecast rate is
  `E(target) / P(target)`: earners growing against a saturating player base.
  With fewer than 3 usable reads — or a flat history, as on launch night —
  the fit is degenerate and the forecast falls back to the last read's rate.
- Spread on the logit scale: a random walk `σ = 0.5 × √(days)` combined in
  quadrature with the fit residual, so a flat history still carries an
  honest, widening band instead of a point forecast.
- Trophies pinned at 100% stay at 100%.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` case in the input directory (in name order) and writes one
`/output/<id>.out` per case: JSON `{"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}]}` — five quantiles per trophy, non-decreasing, 0–100.

## Validation

On the practice cases `p00` / `p01` (flat launch-night history) the program
reproduces the house baseline's practice row for a trophy last at 1.10%
(`[0.8974, 1.012, 1.1, 1.1956, 1.3477]`), validates against the output schema
(55 rows, monotone quantiles, 0–100, ≤ 65,536 bytes), runs in ~0.27 s, and is
byte-identical across repeated runs.

## License

MIT — see [LICENSE](LICENSE).
