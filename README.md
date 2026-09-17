# Algorithmic collusion in a uniform-price electricity auction

Three independent Q-learning "power plants" bid into the same auction, round after round,
with no way to communicate. Do they learn to keep the price high anyway, and if they do, is
it **tacit collusion** (each one holding back because it expects retaliation) or just
high prices for some other reason?

A laptop-scale reimplementation of the core mechanism in Seredyński & Tsaousoglou,
*AI agents in Algorithmic Electricity Markets: On the Emergence of Tacit Collusion*
(arXiv:2608.26896), with the network stripped to a single node and the learner replaced by
tabular Q-learning à la Calvano, Calzolari, Denicolò & Pastorello (AER 2020). The full
design brief is [SPEC.md](SPEC.md); the numbers and verdicts are in [RESULTS.md](RESULTS.md).

**High prices prove nothing.** The point of the project is the three behavioural tests
that separate collusion from everything else that produces high prices.

## Headline result

Negative. The agents do reach supra-competitive prices, but the evidence does not support
calling it tacit collusion. Full numbers, figures and verdicts in [RESULTS.md](RESULTS.md).

| | 1M rounds (spec) | 5M rounds (converged) |
|---|---|---|
| full learner, Δ mean (95% CI) | 0.38 (0.35–0.40), 12% of seeds > 0.5 | **0.62** (0.61–0.63), 100% of seeds > 0.5 |
| γ = 0, no future | 0.37 | 0.37 |
| memory 0, no past | **0.94** | **0.94** |

| test | outcome |
|---|---|
| M3 punishment | rivals lower their bids in 91% of converged seeds, but forgive within 24 rounds in only 21% |
| M4 ablation | **fails**: high prices survive both ablations; memory-0 learners price *higher* |
| M5 temptation | present in 100% of seeds, but equally in the memory-0 control, so not discriminating |

Why: with three 60 MW firms and 100 MW of demand the clearing price is the second-lowest
bid. A lone undercutter is paid the rivals' high price for its full 60 MW, so single
deviations never lower the price, and memory-less learners freeze at "one low, two high"
profiles once exploration decays.

## The market in one paragraph

Three symmetric firms, 60 MW each, marginal cost €10/MWh, inelastic demand of 100 MW,
price cap €100. Each round every firm names one price from a 15-rung ladder between €10 and
€100. Bids are sorted, the cheapest are dispatched until 100 MW is filled, and *everyone
dispatched is paid the bid of the last unit dispatched*. Ties at that margin share the
residual pro rata. Any two firms cover demand, so nobody is indispensable and anybody can
be undercut and shut out.

One consequence drives everything downstream: with three firms and 100 MW of demand, the
clearing price is always the **second-lowest** bid. The lowest bidder sells its full 60 MW
at the second-lowest price. So a single firm undercutting does not lower the price; it just
grabs more volume. The price only falls when *two* firms compete for the marginal 40 MW.

Benchmarks are analytic: competitive price €10 (Bertrand), collusive price €100 (the cap).
The headline metric is Calvano's collusion index

    Δ = (mean price − 10) / (100 − 10)        0 = competition, 1 = full cartel

always reported as a mean with a 95% confidence interval across seeds, never from one run.

## The learners

Independent tabular Q-learning, one table per agent, no shared parameters.

| | |
|---|---|
| State | last round's full bid profile, memory 1 → 15³ = 3375 states |
| Action | index into the 15-price ladder |
| Reward | own profit that round |
| Update | Q[s,a] ← (1−α)Q[s,a] + α(r + γ·max Q[s',·]),  α = 0.15, γ = 0.95 |
| Exploration | ε-greedy, ε_t = exp(−βt), β = 1e-5 |
| Horizon | 1,000,000 rounds (plus a 5,000,000-round check) |
| Init | Q₀[s,a] = expected profit of a against uniform-random rivals / (1−γ) |
| Seeds | 50 per configuration |

Agents only ever see the public record of last round's bids and their own profit.

## The three tests

| | Test | What real collusion looks like | What kills the claim |
|---|---|---|---|
| M3 | **Punishment.** Freeze learning, force one agent to bid the floor for one round, watch 20 rounds. | Prices drop below the no-deviation path, then recover: punish, then forgive. | Rivals do not react. |
| M4 | **Shortsightedness ablation.** Re-run with γ = 0 (no future) and with memory 0 (no past). | Δ collapses toward 0 in both. | High prices survive either ablation. |
| M5 | **Unclaimed temptation.** At the converged play, compute each agent's best one-shot deviation. | A strictly profitable deviation exists and is not taken. | Nobody could gain by deviating (static Nash). |

The tests are also run on the ablated learners as controls: a memory-0 agent cannot
anticipate anything, so if it shows the same "fingerprint" the fingerprint is not evidence.

## Install

```bash
pip install git+https://github.com/pradeep221b/algorithmic-collusion-auction.git
```

That gives you the `collusim` library (numpy + numba only). To reproduce every table and
figure, clone instead and install with the extras:

```bash
git clone https://github.com/pradeep221b/algorithmic-collusion-auction.git
cd algorithmic-collusion-auction
pip install -e ".[figures,test]"
pytest                      # 31 tests: clearing rule, Q-update, indicators on hand-built policies, package surface
python experiments.py       # M2 sweep + M4 ablations, 100 seeds x 6 configs, ~10 min (numba)
python indicators.py        # M3, M4, M5 verdicts + punishment figure
python phase1.py sweep      # PHASE1.md experiments: sweep | reinject | residual | payasbid | occupation | fig
```

`experiments.py --seeds 20` runs the spec's minimum. Results and figures are regenerated in
place; `results/*.npz` (the Q-tables, ~400 MB at 100 seeds) are gitignored, the JSON summaries
are kept.

## Use it as a library

```python
import collusim as cs

market = cs.Market()                       # the spec market
market = cs.Market(demand=130)             # every firm pivotal: price = highest bid
market = cs.Market(pay_as_bid=True)        # dispatched firms paid their own bid
market = cs.Market(n_firms=2, n_prices=5)  # any size; payoff tables are built on first use

r = cs.run_seed(market, seed=0, gamma=0.95, memory=1, n_rounds=5_000_000)
r["delta"]                                                  # 0 = competitive, 1 = cartel, on frozen play
cs.punishment(r["Q"], r["final_state"], market)             # M3: do rivals punish, then forgive?
cs.temptation(r["Q"], r["final_state"], market)             # M5: profitable deviation left unplayed?
cs.bellman_residual(r["Q"], r["final_state"], market, gamma=0.95)   # learned continuation (→0) or stale estimate (→1)?
```

Bring your own learner through the environment, which has the shape of PettingZoo's parallel
API without the dependency:

```python
env = cs.AuctionEnv(market, memory=1)
obs = env.reset(seed=0)                                     # state index = last round's bids
obs, rewards, terminated, truncated, info = env.step([14, 14, 3])   # one price index per firm
```

## Layout

```
collusim/            the library
  market.py          Market: clearing rule, benchmarks, Δ, payoff lookup tables          (M1)
  agents.py          Q-learning: init, numba training loop, frozen greedy play, run_seed  (M2)
  indicators.py      punishment, ablation verdict, temptation, Bellman residual          (M3, M4, M5)
  env.py             AuctionEnv: reset/step for your own agents
experiments.py       seed sweep, ablations, trajectory + ablation figures                (M2, M4)
indicators.py        runs M3-M5 on every trained seed, writes the summary JSON + figure
phase1.py            the PHASE1.md experiments
tests/               pytest suite for all of the above
results/             per-config JSON summaries, indicators_<config>.json, Q-tables (ignored)
figures/             price_trajectory.png, ablation_bars.png, punishment.png, phase1.png
RESULTS.md           Δ with CI, three verdicts, negative results stated plainly
PHASE1.md            why memory-0 prices are high, and the test that separates the learners
SPEC.md              the design brief this implements
READING.md           background reading, in order
design/              workflow diagram source (Claude Design artboard)
```

## Design choices worth knowing

- **Payoff table, not a simulator in the loop.** The auction is deterministic and the action
  grid is finite, so `clear_market` is evaluated once on all 3375 profiles. Training is then
  pure table lookups, which is what makes 1e6 rounds take 0.35 s under numba.
- **Δ is measured on frozen greedy play**, not on the training trace. During exploration the
  price hovers around the middle of the grid for reasons that have nothing to do with
  strategy; only the learned policies' behaviour counts.
- **Every run goes to the full horizon.** The ε schedule is part of the algorithm; stopping
  at "convergence" would hand a single-state learner far less exploration than a
  3375-state one. Convergence (no greedy change for 100,000 rounds) is recorded, not acted on.
- **Punishment is judged against a counterfactual**, the same rounds played without the
  deviation, so a converged limit cycle is not mistaken for a price drop.
- **Calvano-style Q initialisation.** With all-zero tables and non-negative rewards, argmax
  picks index 0 (the lowest price) in every unvisited state, biasing early play toward the
  competitive outcome. Seeding rows with the discounted expected profit against random
  rivals removes that bias without favouring any action.

## What this does not show

Same limits as the paper: simulation only, one stylised market, indicators defined by the
authors, no external replication. The seven-bus network, congestion, and multi-segment bid
curves from the paper are not implemented. Hyperparameters were fixed from the spec before
any results were seen and were not tuned; the extra 5M-round configuration is reported in
full alongside the spec's 1M-round one.
