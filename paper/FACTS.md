# FACTS — the only source of numbers for the paper

Every number below is taken from `results/*.json` (100 seeds per cell unless stated) or computed
directly from the payoff table with `collusim`. Provenance in brackets. Writers cite from here;
verifiers check against here. Nothing else counts.

## 1. Market (SPEC.md; collusim.Market defaults)
- 3 symmetric firms, capacity 60 MW each (total 180), marginal cost €10/MWh, inelastic demand 100 MW,
  price cap €100/MWh. Any two firms cover demand (120 ≥ 100), so no firm is pivotal.
- Action grid: 15 prices, linspace(10, 100): 10, 16.43, 22.86, 29.29, 35.71, 42.14, 48.57, 55.00,
  61.43, 67.86, 74.29, 80.71, 87.14, 93.57, 100. One rung = €6.43.
- Clearing: sort bids ascending, dispatch until 100 MW covered, clearing price = last bid dispatched
  (uniform price). Ties at the margin split pro rata to capacity. With 3×60 vs 100, the clearing
  price is always the SECOND-LOWEST bid: the lowest bidder sells 60 MW, the second-lowest 40 MW,
  the highest 0 (unless tied).
- Consequence, stated precisely [verified with collusim.Market.clear]: from a profile whose TWO
  LOWEST BIDS ARE TIED (which includes every symmetric profile), a lone undercut is paid the rivals'
  price for 60 MW and does NOT move the price — e.g. (100,100,100) → (93.57,100,100) both clear at
  100, the undercutter's profit going 3,000 → 5,400. From such a profile the price falls only when a
  second firm follows. When the two lowest bids DIFFER, an undercut landing between them DOES move
  the price: at the memory-0 rest point (10,100,100) a high bidder shaving one rung gives
  (10,93.57,100), which clears at 93.57 with the undercutter selling 40 MW for 3,343. Never write
  the unqualified form "a lone undercut never moves the price".
- Benchmarks: competitive p = €10 (bid at cost; Δ = 0), collusive p = €100 (cap; Δ = 1).
  Collusion index Δ = (mean clearing price − 10) / 90.
- Pay-as-bid variant (collusim.Market(pay_as_bid=True)): same dispatch, each dispatched firm paid
  its own bid; reported "price" stays the marginal bid so Δ is comparable.
- Pure Nash equilibria of the ONE-SHOT game on the grid [enumerated over all 3375 profiles]: 7
  profiles in 3 shapes: (10,10,10) price 10, profits (0,0,0), Δ 0.00; permutations of (10,16.43,16.43)
  price 16.43, profits (386,129,129), Δ 0.07 [3 profiles]; permutations of (10,22.86,22.86) price
  22.86, profits (771,257,257), Δ 0.14 [3 profiles]. Max Δ of any one-shot Nash = 0.14. The two
  non-zero shapes, (c, c+δ, c+δ) and (c, c+2δ, c+2δ), are one-shot Nash on any grid with rung δ
  [checked by hand from the clearing rule]; a finer grid shrinks their Δ, it does not remove them.
- Best replies at profiles the learners freeze at [computed]:
  (22.86,100,100): firm bidding 22.86 earns 5,400 (best reply: stay); each firm at 100 earns 1,800,
  best reply bid 93.57 earns 3,343. (100,100,100): each earns 3,000 (100/3 MW at margin 90); any
  undercut sells 60 MW at price 100 → 5,400; gap 2,400. (35.71,35.71,35.71): each 857, undercut 1,543.
- Pay-as-bid symmetric ties, one-rung undercut gain [computed]: tie at 67.86: earn 1,929 → 3,086
  (gain 1,157); at 80.71: 2,357 → 3,857 (1,500); at 87.14: 2,571 → 4,243 (1,671); at 93.57:
  2,786 → 4,629 (1,843).

## 2. Learners (collusim.agents; SPEC.md)
- Independent tabular Q-learning, one table per firm, no communication, no shared parameters.
  Update Q[s,a] ← (1−α)Q[s,a] + α(r + γ max_a' Q[s',a']). α = 0.15, γ = 0.95.
- Exploration ε_t = exp(−β t), β = 1e−5 (ε = 0.37 at 1e5 rounds, 0.05 at 3e5, 4.5e−5 at 1e6).
- State (memory 1): last round's full bid profile, 15³ = 3,375 states. Memory 0: one state.
  Agents observe only their own profit and the public record of last round's bids.
- Q initialisation (Calvano et al.): Q₀[s,a] = E[profit(a, rivals uniform)] / (1−γ), so an all-zero
  table does not bias early play toward index 0 (the lowest price).
- Tie-breaks and exclusions [collusim/agents.py _argmax; collusim/indicators.py punishment,
  temptation]: the greedy argmax breaks ties toward the lowest index (lowest price). The M3
  verdict is computed from the rivals' bids only; the deviator's own later bids are excluded.
  M3 "no reaction" is the complement of "punished": rivals' mean bid not more than half a rung
  below the counterfactual in rounds 1–12. In M5, ties among best deviations resolve to the
  highest price (the smallest undercut).
- Horizon T = 1M rounds (spec) and 5M rounds ("long"). Every run goes to the full horizon.
- "Converged" = no agent's greedy action changed during the last 100,000 rounds. Recorded, not
  acted on.
- Δ measured on FROZEN greedy play (learning and exploration off): 1,000 rounds from the final
  state, first 100 dropped, mean clearing price → Δ.
- 100 seeds per cell. CI = mean ± 1.96 × SE across seeds. Hyperparameters fixed before any result
  was seen (from the spec) and never tuned. A 50-seed run of everything preceded the 100-seed run;
  no conclusion changed (Δ moved ≤ 0.02; CIs narrowed ≈ 1/√2).
- Speed: numba-compiled loop, ≈ 0.4 s per 1M rounds per seed. 100 seeds × 6 base configs ≈ 10
  min; whole study a few laptop-hours, no GPU.
- Measured wall-clock at 100 seeds, one process per job, jobs run in parallel on a 12-thread laptop
  [seconds fields in results/*.json; session logs]: the six base configs 68 + 40 + 27 + 166 + 117 +
  102 s ≈ 9 min total; β sweep 10 cells 360 + 485 + 152 + 196 + 94 + 105 + 57 + 78 + 45 + 58 s ≈ 27
  min; pay-as-bid 122 + 135 s ≈ 4 min; ε re-injection ≈ 6 min; Bellman residual ≈ 2 min; M3–M5
  indicators on all six configs ≈ 3 min; occupation curves ≈ 10 min per learner (5 ε values × 100
  seeds × 3M rounds) and ≈ 20 min for the single 30M-round point. Everything: under 1.5 h of wall
  clock with the jobs overlapped, about 2 h sequential. Python 3.14, numpy 2.x, numba. Package `collusim`
  v0.1.0, 31 tests. Repo: https://github.com/pradeep221b/algorithmic-collusion-auction

## 3. M2 — price levels [results/<config>.json summary]
| config | T | Δ mean | 95% CI | median | min–max | seeds Δ>0.5 | converged |
|---|---|---|---|---|---|---|---|
| baseline γ=0.95 m=1 | 1M | 0.375 | 0.354–0.397 | 0.343 | 0.143–0.679 | 12% | 0% |
| long (baseline) | 5M | 0.621 | 0.612–0.630 | 0.625 | 0.518–0.714 | 100% | 100% |
| γ=0 | 1M | 0.373 | 0.353–0.393 | 0.386 | 0.071–0.619 | 5% | 0% |
| γ=0 long | 5M | 0.370 | 0.350–0.390 | 0.386 | 0.071–0.619 | 4% | 100% |
| memory 0 | 1M | 0.943 | 0.931–0.955 | 0.929 | 0.714–1.000 | 100% | 79% |
| memory 0 long | 5M | 0.943 | 0.931–0.954 | 0.929 | 0.714–1.000 | 100% | 100% |
| pay-as-bid m=0 | 5M | 0.845 | 0.819–0.871 | 0.893 | 0.571–1.000 | 100% | 100% |
| pay-as-bid m=1 | 5M | 0.689 | 0.677–0.701 | 0.714 | 0.536–0.857 | 100% | 100% |
- Mean clearing price on frozen play: baseline 1M €43.8; long €65.9; γ=0 long €43.3; memory-0 long €94.9.
- Baseline 1M Δ distribution: 1 seed in [0,0.2), 85 in [0.2,0.5), 14 in [0.5,0.7). No seed converged.
- Long 5M: 99 seeds in [0.5,0.7), 1 in [0.7,1]. Seeds jump one at a time from a mean price near
  €40 to €60–72 and then never move; convergence round median 1.63M, range 1.20M–2.88M.
- γ=0 long: convergence median 1.26M (1.14–1.45M). Memory-0 long: median 0.83M (0.72–0.99M).
- M2 acceptance rule [SPEC.md, M2 "Accept when"]: "a meaningful fraction of seeds converge to Δ > 0.5".
  No numerical threshold is recorded anywhere. Verdict from the rows above [RESULTS.md, summary of
  verdicts]: baseline 1M, 12% of seeds > 0.5 and 0% converged → not met; long 5M, 100% > 0.5 and
  100% converged → met.

### Converged play structure [greedy_play on stored Q-tables, cycle = smallest period of the state sequence]
- Long 5M (memory 1): cycle lengths 1:3, 2:35, 3:21, 4:27, 5:7, 6:5, 7:2 seeds → 86/100 have period ≤ 4.
  The 35 two-cycles alternate a symmetric high profile (mean high price €96.5; bids 94–100) with a
  symmetric low profile (mean low price €41.0; bids 36–42). Most common last-round profiles:
  (100,100,100) ×13, (94,94,94) ×6, (36,42,42) ×6, (36,36,42) ×4, (87,87,87) ×4, (81,81,81) ×3.
  Example 2-cycle: round t bids (100,94,100) → price 100; t+1 bids (36,42,36) → price 36; repeat.
- Baseline 1M (not converged): periods spread from 1 to 56; only 23/100 ≤ 4; the 10 two-cycles have
  high €76.9 / low €34.4.
- γ=0 long: long irregular cycles, 20/100 with period ≤ 4, periods up to 25. Most common frozen
  profiles **from the stored `last_profile` field** (the convention used everywhere in the paper):
  (10,16,16) ×13, (10,42,42) ×10, (10,36,36) ×9, (16,42,42) ×8, (10,29,29) ×8, (10,23,23) ×5.
  CAVEAT, must be stated wherever these are quoted: because these seeds sit on cycles of period up
  to 25, a last-round profile is one snapshot of a cycle, and the counts shift by ~3 seeds per cell
  with the round sampled (a 300-round greedy replay gives 15/10/10/9/7/7). The memory-0 counts are
  fixed points and identical under both conventions; the full learner's top three are too.
- Memory-0 long: 100/100 seeds at a FIXED POINT (period 1), and 100/100 of them exactly of the form
  (low, high, high): one firm low, two firms tied high. Low bidder at €10 (36 seeds), 16.43 (31),
  22.86 (18), 29.29 (11), 42.14 (2), 67.86 (2). High pair at €100 (43), 93.57 (36), 87.14 (20),
  74.29 (1). Most common: (10,100,100) ×21, (16,100,100) ×12, (10,94,94) ×11, (16,87,87) ×10,
  (16,94,94) ×9, (23,94,94) ×8. Clearing price = the high bid (second-lowest); the low bidder sells
  60 MW at it and earns the most; each high bidder sells 20 MW.
- Pay-as-bid frozen profiles [phase1_payasbid + last_profile]: m=0: (68,68,68) ×16, (94,94,94) ×12,
  (81,81,81) ×9, (87,87,87) ×8 — symmetric ties, each sells a third at its own bid.
  m=1: (55,55,55) ×13, (49,49,55) ×9, (61,61,61) ×8, (49,49,49) ×8.

## 4. M3 — punishment [results/indicators_<config>.json]
Procedure: learning frozen; greedy play settles 200 rounds; 6 pre rounds recorded; the deviator is
forced to bid €10 for one round; 24 post rounds of greedy play. Judged on the RIVALS' bids against a
counterfactual path (same rounds, no deviation), in two 12-round windows (cycle lengths 1–4 and 6
all divide 12). punish-forgive: rivals ≥ half a rung (€3.21) below counterfactual in rounds 1–12 AND back
within half a rung in 13–24. punish, no recovery: below in 1–12 and still below in 13–24. no reaction:
rivals do not move. "no-op" = the deviator was already bidding €10.
| config | seeds judged | punish-forgive | punish, no recovery | no reaction | no-op | forgive for ≥1 of the 3 deviators |
|---|---|---|---|---|---|---|
| baseline 1M, collusive seeds only (Δ>0.5) | 12 | 33% | 50% | 17% | 8% | 67% |
| long 5M, all seeds (all collusive) | 100 | 21% | 70% | 9% | 2% | 47% |
| γ=0 1M, collusive seeds | 5 | 0% | 20% | 80% | 40% | 40% |
| γ=0 5M, collusive seeds | 4 | 25% | 25% | 50% | 25% | 50% |
| memory 0, 1M and 5M, all seeds | 100 | 0% | 0% | 100% | 10% / 11% | 0% |
- Long 5M, deviator = firm 1 (code agent 0): rivals react (punished, either verdict) in 91% of seeds.
  Deviator firm 2: forgive 20 / no-recovery 69 / none 11. Deviator firm 3: 17 / 73 / 10. The verdict for a
  GIVEN seed changes with the identity of the deviator in 61 of 100 seeds.
- Long 5M means over all seeds: rivals' bids 16.7 below counterfactual in rounds 1–12 and 13.5 below in
  13–24; clearing price €66.0 before, €45.1 in rounds 1–12 (counterfactual €65.9), €49.4 in 13–24
  (counterfactual €65.9).
  Forgive subset (21 seeds): rival drop 10.9 then 0.2; price €49.9 in rounds 1–12 and €63.2 in 13–24
  vs counterfactual €65.4 (pre-deviation price €65.4).
  No-recovery subset (70 seeds): rival drop 20.5 then 19.1; price €41.8 (1–12) and €43.8 (13–24) vs €66.2.
- The one-round deviation is profitable ONLY at profiles whose two lowest bids are tied, i.e. the
  symmetric ones the full learner converges to (3,000 → 5,400). At a memory-0 rest point
  (low, high, high) a high bidder forced to €10 becomes the lowest bidder, the price falls to the
  LOW BIDDER's bid, and the deviator earns less than before: 0 when the low bidder is at €10 (36 of
  100 seeds), else 386 at 16.43, 772 at 22.86, 1,157 at 29.29 — against 1,800 before [verified].
  The M3 verdict is read from the rivals' bids, so this does not affect it. Original wording, true
  only for the tied case and kept for reference:
- The one-round deviation is PROFITABLE: the deviator sells 60 MW at the rivals' price; the clearing
  price does not move in the deviation round because it is the second-lowest bid.
- Memory-0: nothing to react to (one state), 100% no reaction: the test discriminates.

## 5. M4 — shortsightedness ablation [indicators_baseline/long ablation]
Collapse criterion (fixed in advance): ablated Δ < 0.25 AND < half the full learner's Δ.
| T | full learner | γ=0 | memory 0 | verdict |
|---|---|---|---|---|
| 1M | 0.375 | 0.373 (not collapsed) | 0.943 (not collapsed) | fail |
| 5M | 0.621 | 0.370 (not collapsed) | 0.943 (not collapsed) | fail |
- γ=0 removes about 40% of the converged price premium (0.62 → 0.37), not all of it; at 1M it is
  statistically indistinguishable from the full learner (0.375 vs 0.373).

## 6. M5 — unclaimed temptation [indicators_<config>.json]
At every state on the converged cycle, each firm's best one-shot deviation profit minus realised
profit, rivals fixed. Verdict "unclaimed temptation" if any gap > 0; "static Nash" if all gaps = 0.
| config | seeds with a profitable unplayed deviation | mean best gap €/round (collusive seeds) |
|---|---|---|
| baseline 1M, collusive seeds | 100% (100% of all seeds) | 2,475 |
| long 5M | 100% | 2,226 |
| γ=0 1M / 5M | 100% of collusive (95% of all seeds) | 1,209 / 1,221 |
| memory 0 1M / 5M (control) | 100% | 1,444 / 1,448 |
- The 5% of γ=0 seeds with no temptation sit at one-shot Nash profiles (e.g. (10,16,16)).
- At the all-cap profile each firm earns €3,000 and could earn €5,400 by shaving one rung (gap 2,400).

## 7. Phase 1 — pre-registered predictions and outcomes
| # | experiment | prediction | held? |
|---|---|---|---|
| 1 | β sweep (slower exploration decay) | memory-0 Δ falls toward 0; memory-1 holds | no |
| 2 | ε re-injection from converged tables | memory-1 re-converges to same Δ; memory-0 collapses | half: memory-1 yes; memory-0 also re-converges |
| 3 | Bellman residual on the temptation action | memory-0 large/stale; memory-1 small/learned | yes, with a twist (both stale; consequence differs) |
| 4 | pay-as-bid | (low,high,high) rest point vanishes; memory-0 Δ falls a lot | no: Δ falls 0.10, rest point moves, freeze stays |
| 5 | constant-ε occupation measure | frozen Δ = lim ε→0 of the time-average Δ | yes, for both learners; memory-1 needs ~30× less noise |
Two of five failed.

### 7.1 β sweep [phase1_sweep.json]. Horizon = 10/β + 4M rounds (ε reaches e^−10, then a 4M tail).
| β | rounds | memory 0 Δ (CI) | memory 1 Δ (CI) | memory 1 converged |
|---|---|---|---|---|
| 1e−6 | 14M | 0.949 (0.936–0.962) | 0.721 (0.695–0.746) | 88% |
| 3e−6 | 7.3M | 0.940 (0.928–0.952) | 0.636 (0.623–0.649) | 100% |
| 1e−5 (spec) | 5M | 0.943 (0.931–0.954) | 0.621 (0.612–0.630) | 100% |
| 3e−5 | 4.3M | 0.957 (0.946–0.968) | 0.618 (0.610–0.626) | 100% |
| 1e−4 | 4.1M | 0.949 (0.938–0.961) | 0.593 (0.583–0.603) | 100% |
Memory-0 converged 100% in every cell. β = 1e−6 gives 10× the exploratory rounds of the spec.

### 7.2 ε re-injection [phase1_reinject.json]. Converged 5M tables; restart with ε_t = ε₀·e^{−βt}, β = 1e−5, 2M rounds; freeze; measure.
| source | ε₀ | Δ before → after | seeds dropping > 0.1 | lowest 1k-round mean price during run | low-bidder identity changed |
|---|---|---|---|---|---|
| memory 1 | 0.05 | 0.621 → 0.622 | 4% | €38 | — |
| memory 1 | 0.5 | 0.621 → 0.621 | 5% | €38 | — |
| memory 0 | 0.05 | 0.943 → 0.945 | 11% | €37 | 71% |
| memory 0 | 0.5 | 0.943 → 0.952 | 10% | €37 | 70% |

### 7.3 Bellman residual [phase1_residual.json]
For every temptation on the converged cycle (firm i, state s, played a_i, best unplayed deviation
a_dev): target = profit(a_dev, rivals at their greedy action) + γ·max_a Q_i(s', a); residual =
target − Q_i(s, a_dev). Same for the played action. "backup prefers deviation" = target(a_dev) >
Q_i(s, a_i), i.e. one fresh update would already flip the decision.
| config | residual on played action | residual on a_dev, mean (CI) | share of temptations where one backup prefers a_dev | seeds at 0% / 100% | seeds with ≥1 temptation |
|---|---|---|---|---|---|
| full learner (long 5M) | ≈ 0 (1e−11) | €2,708 (2,521–2,894) | 31% | 12 / 0 | 100 |
| γ=0 long | ≈ 0 | €865 (824–907) | 100% | 0 / 95 | 95 |
| memory 0 long | ≈ 0 | €6,656 (6,093–7,218) | 100% | 0 / 100 | 100 |
- Full learner per-seed distribution of the share: 12 seeds at 0%, 71 in (0,50%], 17 in (50%,100%),
  0 at 100%; maximum single seed 80%. Memory-0 and γ=0: every seed exactly 100%.
- Note for the 5 γ=0 seeds without temptations: they sit at one-shot Nash profiles; not counted.

### 7.4 Pay-as-bid [phase1_payasbid.json]: see §3 table (m=0: 0.845; m=1: 0.689) and §1 for one-rung gains.

### 7.5 Occupation measure [phase1_occupation*.json]. Constant ε (β=0), 3M rounds (30M for ε=1e−4), first third dropped, 100 seeds.
| ε | memory 0 time-avg Δ (CI) | blocks with price > €55 | memory 1 time-avg Δ (CI) | blocks > €55 |
|---|---|---|---|---|
| 0.1 | 0.446 (0.446–0.446) | 28% | 0.336 (0.336–0.336) | 0% |
| 0.03 | 0.645 (0.645–0.645) | 62% | 0.326 (0.325–0.326) | 0% |
| 0.01 | 0.810 (0.809–0.810) | 84% | 0.333 (0.329–0.336) | 4% |
| 0.003 | 0.903 (0.902–0.905) | 95% | 0.408 (0.401–0.415) | 24% |
| 0.001 | 0.934 (0.931–0.937) | 98% | 0.486 (0.477–0.494) | 48% |
| 0.0001 (30M rounds) | not run | | 0.643 (0.635–0.652) | 94% |
| → 0 (frozen, from 7.1 spec cell) | 0.943 | | 0.621 | |
- Mean size of a block-to-block price move (>€5): memory-0 ≈ €28 both directions at ε ≤ 0.03,
  ≈ €22 at ε = 0.1; memory-1 ≈ €10 at ε ≤ 0.003, ≈ €8 at 0.01, ≈ €4–5 at 0.03, none at 0.1.
- Memory-0 meets its frozen value within 0.01 at ε = 0.001. Memory-1 is 0.14 below its frozen
  value at ε = 0.001 and meets it (0.643 vs 0.621) only at ε = 1e−4 — roughly 30× less noise
  (0.003 → 0.0001) for the same closeness (memory-0 is within 5% at ε = 0.003).

## 8. Interpretation (what the paper may claim; nothing stronger)
- Memory-0 high price: a robust attractor of the learning dynamic. Survives 10× slower decay,
  strong perturbation (ε₀ = 0.5 with 70% role swaps), and a change of auction rule. Not a Nash
  equilibrium of the one-shot game (each high bidder has a €1,543 gap). Mechanism: with ε-greedy
  exploration the memory-0 learners run an Edgeworth-type cycle: undercut discovered → undercut
  firm loses volume → undercuts back → price walks DOWN the grid one rung per learning event; near
  the bottom, raising a bid costs nothing (nothing to lose) so the price jumps UP in one event.
  The walk down is a chain of events each needing a specific exploratory draw at a specific time;
  the jump up is one event. The cycle lives near the top; the less noise, the more so; freezing
  samples that distribution. Analogy: stochastic stability (Kandori–Mailath–Rob; Young): the
  states selected as noise vanishes are those that take the most rare draws to leave and the fewest
  to reach. Use that operational phrasing everywhere; do not write "basins under the noisy dynamic".
  Not the folk theorem: no threat, no memory.
- Memory-1: also a small-noise limit of its occupation measure, but ~30× more fragile; a
  coordinated cycle that single random draws break. Its deviation estimates are ALSO stale
  (residual €2,708), but in 69% of temptations the continuation after deviating is low enough
  that a fresh estimate would not change the decision → a learned continuation, the thing tacit
  collusion consists of. Twelve seeds at 0% are pure learned continuation.
- The behavioural battery (M3–M5) and the Δ-ablation cannot tell the two learners apart in this
  market; the Bellman-residual share can (1.00 vs 0.31, non-overlapping). It needs the Q-table, so
  it is a simulation-side tool, not a screen for real bid data.
- Honest verdict: agents converge to supra-competitive prices under these parameters; the
  standard evidence does not support calling it tacit collusion; part of what the memory-1 learner
  does is learned continuation; the memory-0 price is a property of ε-greedy learning on a
  discrete grid in a non-pivotal uniform-price auction.

## 8b. What was fixed in advance, and what was not [git history: SPEC.md and indicators.py at commit 553760c]
This is the honest account; the paper must not claim more.
- **In SPEC.md, written before any code**: the market, the learners, every hyperparameter (α, γ, β,
  grid, horizons), the four measurements, and QUALITATIVE acceptance rules. M2: "a meaningful
  fraction of seeds converge to Δ > 0.5" (no numeric threshold anywhere). M3: "prices drop below the
  pre-deviation level for several rounds, then recover", judged from a plot of 20 rounds, deviator
  agent 0. M4: "Δ collapses toward 0 in both" (no numeric threshold). M5: "at least one agent's best
  deviation is strictly profitable yet unplayed".
- **Fixed at implementation, before any reported run, but not in SPEC.md**: the M4 collapse rule
  (Δ < 0.25 AND < half the full learner's Δ) and the whole M3 operationalisation — judging on the
  RIVALS' bids against a no-deviation counterfactual, two 12-round windows, half-rung (€3.21)
  threshold. Both are present in `indicators.py` at the first commit (553760c).
- **Chosen with knowledge of the runs**: the 12-round window length is justified by the converged
  cycle lengths (12 divides 1–4 and 6), which were only known after the first runs. The rival-bid
  criterion replaced a price-based one after the price-based version produced false positives on
  converged cycles (a converged cycle's own price dip was read as punishment). Every M3 verdict
  reported here was computed with the final criterion.
- **Phase 1 is not in SPEC.md at all.** Each experiment's prediction was written into the
  `phase1.py` docstring before that experiment ran, but the five experiments were designed
  sequentially, each in response to the previous result.
- **Post-hoc addition**: the ε = 1e−4 / 30M-round occupation point was added after the memory-1
  curve was seen still rising at ε = 1e−3. It is the only cell chosen after seeing the quantity it
  reports.
- Seeds: the 50-seed pilot used `range(50)`, the reported runs `range(100)`; the first 50 seeds are
  the same seeds.

## 8c. Mechanical facts [computed from collusim and results/*.json]
- Trajectory figures: mean clearing price per block of `BLOCK = 1000` rounds. Convergence window
  `CONV_ROUNDS = 100000`. `t_conv` = first round at which no agent's greedy action had changed for
  100,000 rounds, else −1.
- Baseline 1M Δ distribution, exactly: 12 seeds strictly above 0.5, **2 seeds exactly at Δ = 0.500**
  (frozen mean price €55.00), so 14 seeds in [0.5, 0.7). The "12%" and the "14 in [0.5,0.7)" figures
  are both correct and refer to different things.
- Baseline 1M mean trajectory: starts at €54.9, first block at or below €40 at 120k rounds, minimum
  €37.5 at 196k rounds, ends at €43.9.
- Memory-0 after ε re-injection at ε₀ = 0.5: **100 of 100 seeds** end again at a (low, high, high)
  profile (with the roles reassigned in 70% of seeds).
- M3 verdict COUNTS behind the shares [indicators_<config>.json per_seed, deviator = firm 1]:
  baseline 1M, 12 collusive seeds: forgive 4, no recovery 6, no reaction 2, no-op 1, forgive for at
  least one deviator 8. Long 5M, 100 seeds: forgive 21, no recovery 70, no reaction 9, no-op 2,
  forgive for at least one deviator 47. γ=0 1M, 5 collusive: 0 / 1 / 4, no-op 2. γ=0 5M, 4
  collusive: 1 / 1 / 2, no-op 1. Memory-0 1M and 5M, 100 each: 0 / 0 / 100, no-op 10 and 11.
- Representative seed of each punishment figure = the seed with the highest Δ in that configuration
  [indicators_<config>.json "representative_seed"]: long 5M → seed 19; baseline 1M → seed 1;
  γ=0 → 36; memory-0 → 0.

## 9. Limits (must appear)
- Simulation only; one stylised single-node market; no network/congestion, no multi-segment bid
  curves, no demand variation; symmetric firms; tabular learners only (no DQN).
- Hyperparameters not swept except β and the horizon; α, γ, grid size from the spec.
- Punishment windows/thresholds fixed (12-round windows; half-rung); 5- and 7-cycles (9 seeds in
  the long run) can be misread by 12-round windows.
- Bellman residual uses a one-step backup with the current table; Q(s') may itself be stale; a
  k-step rollout would be stricter (not done).
- Occupation curve for memory-1 has one point at ε = 1e−4; the transition is not located precisely.
- No theory: the Edgeworth-cycle argument is verbal.
- The two extra one-shot Nash equilibria (Δ 0.07, 0.14) mean the "competitive benchmark" on the
  grid is a band [0, 0.14], not a point; all reported Δ are far above it.
