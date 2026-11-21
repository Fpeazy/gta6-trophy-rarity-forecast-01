# GTA VI Trophy Rarity Forecast — AdaptiveRefreshEnsemble_v2 + ExactPosteriorFrozen

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fpeazy/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fpeazy/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.

Bounty: <https://make.fun/bounty/gta6-trophy-rarity-forecast-01>

## Method

Adaptive refresh ensemble v2 with an exact-posterior frozen branch. Exophase
prints each rate to 2 decimals and refreshes its counts on its own schedule,
so a refresh that adds many fresh players dilutes every rate down. The
program therefore models the two count curves separately — earners E(t) and
players tracked P(t) — and takes the rate as E(t)/P(t), so dilution falls
out automatically when P grows faster than E. Which branch runs depends on
whether the history ever moved:

1. **Frozen history** (one unique snapshot, or nothing ever moved — the
   practice cases): predict the refresh itself, with the EXACT posterior
   over the gain factor. Launch night's one refresh took P 638 → 3,868
   (×6.06) while earners grew only ~×1.22, so the gain factor G is taken
   as Uniform[1.08, 1.32] (centred on the inferred 1.22, wide enough to
   cover the per-trophy gain spread plus 2-decimal rate quantisation
   noise): P_new = P_last × 6.06, base_i = 100 × E_old,i / P_new, and
   q_τ = base_i × (1.08 + 0.24 × τ). This replaces the v2 Gaussian
   approximation (σ = 0.35), which was far too wide.
2. **Moving history**: an ensemble of a counts fit (log E and log P against
   `ln(1 + hours)`, slopes clamped to [0, 3] and shrunk 50% toward the tier
   median, rate capped per tier: platinum 2%, gold 5%, silver 25%, bronze
   90%) and a logit-trend fit, averaged 50/50 in logit space, then shrunk
   25% toward the last observed rate (persistence), then the median shifted
   DOWN 0.55 logit — a late-refresh dilution prior (v2 change vs v1: a
   refresh is near-certain inside the 4.5-day final window, and the
   unshifted median ran too high after the final dilution).
3. Spread on the logit scale combines a horizon term, the log-E fit
   residual, a rare-trophy floor, P-jump volatility (CV of the P increases),
   and a floor from the observed refresh jumps (half the median |Δlogit|
   across trophies at each P jump over 5%), with a minimum σ of 0.30.
   A trophy at 100% stays pinned at 100.

On the coordinator's synthetic holdout (5 worlds: flat, post-refresh,
steady/weekend-spike/late-refresh 7-day): losses 1.33 / 0.027 / 0.31 /
0.29 / 0.31 — worst case 0.314, beating the house baseline in all four
realistic 7-day worlds (baseline 0.894 / 0.431 / 0.373 / 0.326), and
90% coverage 0.89 on the refresh world, 0.98–1.00 on the 7-day worlds.
(The v2 Gaussian frozen branch scored 0.079 on the refresh world; the
exact posterior cuts it to 0.027.)

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
