# Algorithmic Collusion in a Uniform-Price Electricity Auction

Reimplementation of the core mechanism from **Seredyński & Tsaousoglou, "AI agents in Algorithmic
Electricity Markets: On the Emergence of Tacit Collusion"** (arXiv:2608.26896, 27 Aug 2026),
stripped to a single-node market so it runs on a laptop CPU.

No code was released with the paper. Method reference for the pricing-game analogue:
Calvano, Calzolari, Denicolò & Pastorello (2020), *AER* 110(10) — replication at
https://github.com/matteocourthoud/Algorithmic-Collusion-Replication

## Goal

Show that independent Q-learning agents bidding into a repeated auction converge to
supra-competitive prices with **no communication channel**, then prove it is genuine tacit
collusion using three behavioural tests rather than inferring it from high prices.

The three tests are the point of the project. Getting high prices is easy and proves nothing.

## What was deliberately cut from the paper

- The seven-bus DC-OPF network and locational marginal prices. Single node instead.
- Piecewise-linear multi-segment bid curves. One price per agent per round instead.
- Their specific MARL algorithm (deferred to a section not fully specified). Tabular Q-learning
  instead, which is the standard vehicle for this result and is fully reproducible.

Congestion is where the paper's scenario variation comes from. It is a second-phase extension,
not part of the initial build.

## Market model

Uniform-price auction, single node, repeated indefinitely against the same opponents.

| Parameter | Value | Rationale |
|---|---|---|
| Firms `n` | 3 | Must be ≥3 so no firm is pivotal |
| Capacity `q_i` | 60 MW each | Total 180 |
| Marginal cost `c_i` | €10/MWh each | Symmetric — keeps benchmarks analytic |
| Demand `D` | 100 MW, inelastic | Any two firms cover it (120 ≥ 100) → nobody pivotal |
| Price cap `P_max` | €100/MWh | Required: without a cap the collusive price is unbounded |
| Action grid `A` | 15 prices, linspace(10, 100) | €10.00, €16.43, … €100.00 |

**Non-pivotality matters.** If a single firm were required to meet demand it could name any price
and there is nothing to learn. With `D=100` and three 60 MW firms, any one firm can be excluded,
so undercutting is always a live threat. That threat is what the punishment strategy has to
suppress.

### Clearing rule

1. Collect one price bid per firm.
2. Sort ascending, dispatch greedily until demand is met.
3. Clearing price = bid of the last (most expensive) unit dispatched. Everyone dispatched is paid
   this price, not their own bid.
4. Ties at the marginal price: split the residual demand **pro rata** among tied firms.
5. Profit for firm `i`: `(p_clear - c_i) * q_dispatched_i`.

Pro-rata tie-breaking is not cosmetic. It is what makes matching a rival's price a stable
equilibrium instead of a knife-edge, and undercutting discretely profitable.

### Benchmarks — compute both before writing any learning code

- **Competitive**: `p_comp = €10/MWh` (marginal cost). The static Nash equilibrium of the one-shot
  game, by the Bertrand undercutting argument.
- **Collusive**: `p_coll = €100/MWh` (price cap). Joint-profit maximum under inelastic demand.

**Collusion index** (Calvano's Δ), the single headline number:

```
Δ = (mean_price - p_comp) / (p_coll - p_comp)
```

Δ ≈ 0 is competition. Δ ≈ 1 is full collusion. Published pricing-game results land around
0.7–0.9. Report Δ with a confidence interval across seeds, never from one run.

## Learning algorithm

Independent tabular Q-learning, one Q-table per agent, no shared parameters, no communication.

- **State**: previous round's full action profile — memory 1. `|S| = 15^3 = 3375`.
- **Action**: index into the 15-price grid.
- **Reward**: own realised profit that round.
- **Q-table**: `3375 x 15` per agent. Trivial memory.
- **Update**: `Q[s,a] ← (1-α)Q[s,a] + α(r + γ max_a' Q[s',a'])`
- `α = 0.15`, `γ = 0.95`
- **Exploration**: ε-greedy, `ε_t = exp(-β t)`, `β = 1e-5`
- **Horizon**: 1e6 rounds. Convergence check: no agent's greedy policy changes for 1e5
  consecutive rounds.
- **Seeds**: 20+ minimum. Outcomes are genuinely multi-modal — some seeds collude, some do not.
  A single run tells you nothing and this is the most common way to fool yourself here.

Agents observe only their own profit and the public clearing price. They never see rivals' bids
directly — those enter only through the state, which is the public record of what was bid. This
is the paper's "imperfect public monitoring".

## Milestones

### M1 — Market and benchmarks
`clear_market(bids) -> (price, dispatch_vector)` plus analytic `p_comp` and `p_coll`.

*Accept when*: unit tests pass for all-equal bids, strict undercut, tie at the margin, and a bid
above cap. Verify pro-rata splitting sums to exactly `D`.

### M2 — Learning loop
Three independent Q-learners, run to convergence, plot mean price per 1000 rounds against both
benchmarks. Report Δ across seeds.

*Accept when*: a meaningful fraction of seeds converge to Δ > 0.5. If every seed sits at Δ ≈ 0,
the bug is almost always in the reward or the state encoding — check that `s'` is the profile
just played, not the one before.

### M3 — Indicator 1: punishment (the money plot)
Freeze learning. Force agent 0 to play the lowest price for exactly one round. Resume greedy play
for all agents. Log 20 rounds of prices and profits.

*Accept when*: the plot shows prices drop below the pre-deviation level for several rounds, then
recover — punish-then-forgive. **If prices do not react, it was never collusion.** That negative
result is a legitimate finding and must be reported as such, not tuned away.

### M4 — Indicator 2: shortsightedness ablation
Re-run M2 twice: once with `γ = 0`, once with memory-0 (state is a constant, so no history).

*Accept when*: Δ collapses toward 0 in both. Collusion requires caring about the future and
remembering the past. If high prices survive either ablation, they had some other cause and your
M2 result is not collusion.

### M5 — Indicator 3: unclaimed temptation
At the converged state, compute each agent's one-shot deviation profit — best single-round payoff
from unilaterally changing price, holding rivals fixed.

*Accept when*: at least one agent's best deviation is strictly profitable yet unplayed. That gap
is the fingerprint: money left on the table because retaliation is anticipated.

## Deliverables

- `market.py` — clearing, benchmarks, tests
- `agents.py` — Q-learning
- `experiments.py` — M2 sweep, M4 ablations
- `indicators.py` — the three tests
- `figures/` — price trajectory, punishment plot, ablation bars
- `RESULTS.md` — Δ with CI, all three indicator verdicts, and any negative results stated plainly

## Rules for this project

- Report negative results. "Agents did not collude under these parameters" is a real finding.
- Never report Δ from a single seed.
- Never tune hyperparameters until collusion appears, then present it as a discovery. If you sweep,
  report the whole sweep.
- The paper is simulation-only on one stylised network, unreproduced, with author-defined
  indicators. This project inherits every one of those limits. Any writeup says so.
