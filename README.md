# GTA VI Trophy Rarity Forecast — DualCurve_RefreshDilution_v1

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fpeazy/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fpeazy/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.

Bounty: <https://make.fun/bounty/gta6-trophy-rarity-forecast-01>

## Method

Counts model, not a rate trend. Exophase prints each rate to 2 decimals and
refreshes its counts on its own schedule, so a refresh that adds many fresh
players dilutes every rate down. The program therefore models the two count
curves separately — earners E(t) and players tracked P(t) — and takes the
rate as E(t)/P(t), so dilution falls out automatically when P grows faster
than E.

1. Deduplicate consecutive identical reads (a frozen page carries no signal).
2. Reconstruct integer earners from the rounded rates (interval from the
   2-decimal rounding, nearest to the printed rate, monotone in time, and
   allocated so each snapshot's sum matches `trophies_earned`).
3. Fit `log P` and `log E_i` against `ln(1 + hours since launch)` by least
   squares; each trophy's slope shrinks toward its tier median, and the
   forecast rate is capped per tier (platinum 2%, gold 5%, silver 25%,
   bronze 90%).
4. When fewer than 3 distinct snapshots exist (the practice cases: the page
   never moved), fall back honestly — median = last rate, spread =
   `0.5 * sqrt(horizon in days)`.
5. Quantile spread on the logit scale combines a random-walk term, the fit
   residual, a rare-trophy floor, and refresh-jump uncertainty (CV of the
   P jumps between refreshes). A trophy at 100% stays pinned at 100.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network, fully deterministic (two runs
are byte-identical). One output file per case: `/output/<id>.out`, a JSON
object `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

## License

MIT — see [LICENSE](LICENSE).

---

Optional: earn referral rewards, a second way to make money —
[play on make.fun with referral code afc5cnfr](https://make.fun/?ref=afc5cnfr).
