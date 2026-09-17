"""Tests that separate tacit collusion from merely high prices. All take a trained Q and a Market.

    punishment(Q, state, market)               M3  freeze, force one undercut, do the RIVALS punish, then forgive?
    ablation(baseline, gamma0, memory0)        M4  did Delta collapse without a future / without a past?
    temptation(Q, state, market)               M5  a profitable one-shot deviation exists but is not taken
    bellman_residual(Q, state, market, gamma)      is the unplayed deviation's Q-value learned, or stale?

The first three follow Calvano et al. (2020). The fourth is the Q-table test from PHASE1.md:
memory-0 and gamma=0 learners sit at 1.0 (a single Bellman backup would already flip the decision),
the full learner sits near 0.3 (the continuation after deviating is genuinely worse).
"""
import numpy as np

from . import agents as ag
from .market import encode

VERDICTS = ("punish_forgive", "punish_no_recovery", "no_reaction")


# --- M3 -------------------------------------------------------------------------
def punishment(Q, state, market, deviator=0, warmup=200, pre=6, post=24):
    """Freeze learning, let greedy play settle, force `deviator` to the lowest price for one
    round, then resume greedy play for `post` rounds.

    Judged on the RIVALS' bids against a counterfactual path (same rounds, no deviation):
    the deviator's own later choices must not count as retaliation, and a converged limit
    cycle must not be mistaken for a drop. The 24 post rounds are split into two 12-round
    windows (12 is divisible by cycle lengths 1-4 and 6) and compared as means.
      punish_forgive     rivals bid lower than the counterfactual in rounds 1-12, back by 13-24
      punish_no_recovery rivals bid lower and stay lower (grim trigger, or no route back)
      no_reaction        rivals' bids do not move -> this was never collusion
    """
    profit, price = market.tables
    prices = market.prices
    grid_step = float(prices[1] - prices[0])            # one rung of the price ladder
    settled = ag.greedy_play(Q, profit, price, state, warmup)
    s0 = settled["final_state"]
    n_rounds = pre + 1 + post
    base = ag.greedy_play(Q, profit, price, s0, n_rounds)
    play = ag.greedy_play(Q, profit, price, s0, n_rounds, force={pre: {deviator: 0}})
    rivals = [i for i in range(Q.shape[0]) if i != deviator]
    half = post // 2
    early, late = slice(pre + 1, pre + 1 + half), slice(pre + 1 + half, pre + 1 + post)

    def rival_bid(p, window):
        return float(prices[p["actions"][window][:, rivals]].mean())

    drop_early = rival_bid(base, early) - rival_bid(play, early)   # > 0: rivals bid lower than they would have
    drop_late = rival_bid(base, late) - rival_bid(play, late)
    punished = bool(drop_early > grid_step / 2)
    recovered = bool(drop_late < grid_step / 2)
    verdict = "punish_forgive" if punished and recovered else "punish_no_recovery" if punished else "no_reaction"
    return dict(
        verdict=verdict, punished=punished, recovered=recovered,
        deviation_was_noop=bool(base["actions"][pre, deviator] == 0),   # deviator already bid the floor
        rival_drop_early=drop_early, rival_drop_late=drop_late,
        p_pre=float(base["prices"][:pre].mean()),
        p_early=float(play["prices"][early].mean()), p_cf_early=float(base["prices"][early].mean()),
        p_late=float(play["prices"][late].mean()), p_cf_late=float(base["prices"][late].mean()),
        rounds=list(range(-pre, post + 1)), prices=play["prices"].tolist(),
        counterfactual=base["prices"].tolist(), bids=prices[play["actions"]].tolist(),
        profits=play["profits"].tolist(), deviator=deviator,
    )


# --- M4 -------------------------------------------------------------------------
def ablation(baseline, gamma0, memory0, threshold=0.25):
    """Delta has 'collapsed' if the ablated mean is below `threshold` AND below half the baseline."""
    def collapsed(s):
        return bool(s["mean"] < threshold and s["mean"] < 0.5 * baseline["mean"])
    out = {"baseline": baseline["mean"], "gamma0": gamma0["mean"], "memory0": memory0["mean"],
           "gamma0_collapsed": collapsed(gamma0), "memory0_collapsed": collapsed(memory0)}
    out["verdict"] = "pass" if out["gamma0_collapsed"] and out["memory0_collapsed"] else "fail"
    return out


# --- M5 -------------------------------------------------------------------------
def temptation(Q, state, market, rounds=100):
    """One-shot deviation gap at every state on the converged cycle, per agent.

    gap_i = max_b profit_i(b, rivals) - profit_i(played) with rivals held fixed.
    gap > 0 and not taken = money left on the table because retaliation is anticipated.
    gap == 0 everywhere = the play is a static Nash equilibrium: nothing to explain.
    """
    profit, price = market.tables
    play = ag.greedy_play(Q, profit, price, state, rounds)
    n, _, k = Q.shape
    gaps = np.zeros((rounds, n))
    best_dev = np.zeros((rounds, n), dtype=int)
    for t, a in enumerate(play["actions"]):
        for i in range(n):
            alts = np.array([profit[encode([*a[:i], b, *a[i + 1:]], k), i] for b in range(k)])
            best_dev[t, i] = int(np.flatnonzero(alts == alts.max()).max())   # ties: the smallest undercut
            gaps[t, i] = alts.max() - profit[encode(a, k), i]
    gap_max = gaps.max(axis=0)
    return dict(
        verdict="unclaimed_temptation" if gap_max.max() > 1e-9 else "static_nash",
        gap_mean=gaps.mean(axis=0).tolist(), gap_max=gap_max.tolist(),
        played_profit=profit[encode(play["actions"][-1], k)].tolist(),
        played_action=play["actions"][-1].tolist(), best_deviation=best_dev[-1].tolist(),
        frac_rounds_static_nash=float((gaps.max(axis=1) <= 1e-9).mean()),
    )


# --- Learned or stale? -----------------------------------------------------------
def bellman_residual(Q, state, market, gamma, rounds=100):
    """For every temptation on the converged cycle: target = r(a_dev, rivals greedy) + gamma * max Q[s'],
    residual = target - Q[s, a_dev]. Same for the played action (always ~0, by construction).

    frac_backup_prefers_dev: share of temptations where one fresh backup would already make the
    deviation look better than what is played. 1.0 = the low estimate is inherited from the
    exploration era and nobody has looked since; near 0 = the state reached by deviating is
    genuinely worth less, i.e. a learned continuation.
    """
    profit, price = market.tables
    play = ag.greedy_play(Q, profit, price, state, rounds)
    n, S, k = Q.shape
    res_dev, res_played, dev_better = [], [], []
    for t, a in enumerate(play["actions"]):
        s = play["states"][t]
        for i in range(n):
            alts = np.array([profit[encode([*a[:i], b, *a[i + 1:]], k), i] for b in range(k)])
            b = int(np.flatnonzero(alts == alts.max()).max())
            if alts[b] - alts[a[i]] <= 1e-9:
                continue                                   # no temptation here
            for act, store in ((b, res_dev), (a[i], res_played)):
                idx = encode([*a[:i], act, *a[i + 1:]], k)
                nxt = idx if S > 1 else 0
                target = profit[idx, i] + gamma * Q[i, nxt].max()
                store.append(target - Q[i, s, act])
            dev_better.append(res_dev[-1] + Q[i, s, b] > Q[i, s, a[i]])
    f = lambda x: float(np.mean(x)) if x else float("nan")
    return dict(residual_dev=f(res_dev), residual_played=f(res_played),
                frac_backup_prefers_dev=f(dev_better), n_temptations=len(res_dev))
