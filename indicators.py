"""M3-M5: the three behavioural tests that separate tacit collusion from merely high prices.

    python indicators.py        # after experiments.py; reads results/<config>/*.npz

M3 punishment    - freeze learning, force one undercut, watch whether RIVALS punish then forgive
M4 ablation      - Delta must collapse without a future (gamma=0) and without a past (memory 0)
M5 temptation    - at the converged play, a profitable one-shot deviation exists but is not taken

Writes results/indicators_<config>.json and figures/punishment*.png, prints the verdicts.
The ablations are run through M3/M5 as controls.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import agents as ag
import market as m
from experiments import COLORS, FIGURES, HORIZONS, PRICE, PROFIT, RESULTS, load_summary, style

GRID_STEP = float(m.PRICES[1] - m.PRICES[0])   # one rung of the price ladder, EUR 6.43
COLLUSIVE = 0.5                                  # a seed "colluded" if its Delta exceeds this
VERDICTS = ("punish_forgive", "punish_no_recovery", "no_reaction")


def load_run(name, seed):
    z = np.load(RESULTS / name / f"seed_{seed}.npz")
    return z["Q"], int(z["final_state"])


# --- M3 -------------------------------------------------------------------------
def punishment(Q, state, deviator=0, warmup=200, pre=6, post=24):
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
    settled = ag.greedy_play(Q, PROFIT, PRICE, state, warmup)
    s0 = settled["final_state"]
    n_rounds = pre + 1 + post
    base = ag.greedy_play(Q, PROFIT, PRICE, s0, n_rounds)
    play = ag.greedy_play(Q, PROFIT, PRICE, s0, n_rounds, force={pre: {deviator: 0}})
    rivals = [i for i in range(Q.shape[0]) if i != deviator]
    half = post // 2
    early, late = slice(pre + 1, pre + 1 + half), slice(pre + 1 + half, pre + 1 + post)

    def rival_bid(p, window):
        return float(m.PRICES[p["actions"][window][:, rivals]].mean())

    drop_early = rival_bid(base, early) - rival_bid(play, early)   # > 0: rivals bid lower than they would have
    drop_late = rival_bid(base, late) - rival_bid(play, late)
    punished = bool(drop_early > GRID_STEP / 2)
    recovered = bool(drop_late < GRID_STEP / 2)
    verdict = "punish_forgive" if punished and recovered else "punish_no_recovery" if punished else "no_reaction"
    return dict(
        verdict=verdict, punished=punished, recovered=recovered,
        deviation_was_noop=bool(base["actions"][pre, deviator] == 0),   # deviator already bid the floor
        rival_drop_early=drop_early, rival_drop_late=drop_late,
        p_pre=float(base["prices"][:pre].mean()),
        p_early=float(play["prices"][early].mean()), p_cf_early=float(base["prices"][early].mean()),
        p_late=float(play["prices"][late].mean()), p_cf_late=float(base["prices"][late].mean()),
        rounds=list(range(-pre, post + 1)), prices=play["prices"].tolist(),
        counterfactual=base["prices"].tolist(), bids=m.PRICES[play["actions"]].tolist(),
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
def temptation(Q, state, rounds=100):
    """One-shot deviation gap at every state on the converged cycle, per agent.

    gap_i = max_b profit_i(b, rivals) - profit_i(played) with rivals held fixed.
    gap > 0 and not taken = money left on the table because retaliation is anticipated.
    gap == 0 everywhere = the play is a static Nash equilibrium: nothing to explain.
    """
    play = ag.greedy_play(Q, PROFIT, PRICE, state, rounds)
    n, _, k = Q.shape
    gaps = np.zeros((rounds, n))
    best_dev = np.zeros((rounds, n), dtype=int)
    for t, a in enumerate(play["actions"]):
        for i in range(n):
            alts = np.array([PROFIT[m.encode([*a[:i], b, *a[i + 1:]]), i] for b in range(k)])
            best_dev[t, i] = int(np.flatnonzero(alts == alts.max()).max())   # ties: the smallest undercut
            gaps[t, i] = alts.max() - PROFIT[m.encode(a), i]
    gap_max = gaps.max(axis=0)
    return dict(
        verdict="unclaimed_temptation" if gap_max.max() > 1e-9 else "static_nash",
        gap_mean=gaps.mean(axis=0).tolist(), gap_max=gap_max.tolist(),
        played_profit=PROFIT[m.encode(play["actions"][-1])].tolist(),
        played_action=play["actions"][-1].tolist(), best_deviation=best_dev[-1].tolist(),
        frac_rounds_static_nash=float((gaps.max(axis=1) <= 1e-9).mean()),
    )


# --- Figure ---------------------------------------------------------------------
def fig_punishment(per_seed, representative, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    r = per_seed[representative]["punishment"][0]
    x = r["rounds"]
    for i, c in enumerate(("s1", "s2", "s3")):
        ax1.plot(x, np.array(r["bids"])[:, i], color=COLORS[c], linewidth=1.2, alpha=0.9,
                 marker="o", markersize=3, label=f"bid, agent {i}")
    ax1.plot(x, r["prices"], color=COLORS["ink"], linewidth=2.2, label="clearing price")
    ax1.axvline(0, color=COLORS["muted"], linestyle="--", linewidth=1)
    ax1.text(0.3, 4, "agent 0 forced to €10", fontsize=8, color=COLORS["secondary"])
    ax1.set_title(f"M3  {name}, seed {representative} (highest Δ)", loc="left", fontsize=11, color=COLORS["ink"])
    ax1.set_xlabel("rounds after deviation", color=COLORS["secondary"])
    ax1.set_ylabel("€/MWh", color=COLORS["secondary"])
    ax1.set_ylim(0, 108)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax1)

    coll = [s for s in per_seed if per_seed[s]["delta"] > COLLUSIVE]
    comp = [s for s in per_seed if per_seed[s]["delta"] <= COLLUSIVE]
    for seeds, color, label in ((coll, "ink", f"collusive seeds (Δ>0.5), n={len(coll)}"),
                                (comp, "muted", f"non-collusive seeds, n={len(comp)}")):
        if not seeds:
            continue
        P = np.array([per_seed[s]["punishment"][0]["prices"] for s in seeds])
        C = np.array([per_seed[s]["punishment"][0]["counterfactual"] for s in seeds])
        ax2.plot(x, P.mean(axis=0), color=COLORS[color], linewidth=2, label=label)
        ax2.plot(x, C.mean(axis=0), color=COLORS[color], linewidth=1, linestyle=":", label="  … no-deviation counterfactual")
    ax2.axvline(0, color=COLORS["muted"], linestyle="--", linewidth=1)
    ax2.set_title("M3  mean clearing price across seeds", loc="left", fontsize=11, color=COLORS["ink"])
    ax2.set_xlabel("rounds after deviation", color=COLORS["secondary"])
    ax2.set_ylim(0, 108)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax2)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / ("punishment.png" if name == "baseline" else f"punishment_{name}.png"))
    plt.close(fig)


# --- Driver ---------------------------------------------------------------------
def main(name="baseline", figure=True):
    """Run all three indicators on every seed of configuration `name`."""
    data = load_summary(name)
    per_seed = {}
    for row in data["seeds"]:
        Q, state = load_run(name, row["seed"])
        per_seed[row["seed"]] = dict(
            delta=row["delta"],
            punishment=[punishment(Q, state, deviator=i) for i in range(m.N_FIRMS)],   # [0] = SPEC's agent 0
            temptation=temptation(Q, state),
        )
    coll = [s for s, r in per_seed.items() if r["delta"] > COLLUSIVE]

    def share(seeds, pred):
        return float(np.mean([pred(per_seed[s]) for s in seeds])) if seeds else None

    summary = {
        "n_seeds": len(per_seed), "n_collusive": len(coll),
        "punishment_agent0": {v: share(coll, lambda r, v=v: r["punishment"][0]["verdict"] == v) for v in VERDICTS},
        "punishment_any_agent": share(coll, lambda r: any(p["verdict"] == "punish_forgive" for p in r["punishment"])),
        "punishment_noop_agent0": share(coll, lambda r: r["punishment"][0]["deviation_was_noop"]),
        "temptation_collusive": share(coll, lambda r: r["temptation"]["verdict"] == "unclaimed_temptation"),
        "temptation_all": share(list(per_seed), lambda r: r["temptation"]["verdict"] == "unclaimed_temptation"),
        "mean_gap_collusive": share(coll, lambda r: max(r["temptation"]["gap_max"])),
    }
    for horizon, (full, g0, m0) in HORIZONS.items():
        if name == full and all((RESULTS / f"{c}.json").exists() for c in (g0, m0)):
            summary["ablation"] = ablation(data["summary"], load_summary(g0)["summary"], load_summary(m0)["summary"])
            summary["ablation"]["horizon"] = horizon
    representative = max(per_seed, key=lambda s: per_seed[s]["delta"])
    if figure:
        fig_punishment(per_seed, representative, name)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"indicators_{name}.json").write_text(json.dumps(
        {"summary": summary, "representative_seed": representative, "per_seed": per_seed}, indent=1),
        encoding="utf-8")

    # console output is ASCII-only: Windows consoles choke on Greek letters
    f2 = lambda x: "n/a" if x is None else f"{x:.2f}"
    print(f"[{name}] seeds: {summary['n_seeds']}, collusive (Delta>{COLLUSIVE}): {summary['n_collusive']}")
    print("M3 punishment (agent 0 deviates), share of collusive seeds: "
          + ", ".join(f"{k}={f2(v)}" for k, v in summary["punishment_agent0"].items())
          + f"; deviation was a no-op in {f2(summary['punishment_noop_agent0'])}; "
          f"punish-forgive for at least one deviator: {f2(summary['punishment_any_agent'])}")
    if "ablation" in summary:
        a = summary["ablation"]
        print(f"M4 ablation @{a['horizon']}: full Delta={a['baseline']:.3f}, gamma=0 Delta={a['gamma0']:.3f} "
              f"({'collapsed' if a['gamma0_collapsed'] else 'NOT collapsed'}), memory-0 Delta={a['memory0']:.3f} "
              f"({'collapsed' if a['memory0_collapsed'] else 'NOT collapsed'}) -> {a['verdict']}")
    print(f"M5 temptation: unclaimed profitable deviation in {f2(summary['temptation_collusive'])} of collusive seeds "
          f"({f2(summary['temptation_all'])} of all); mean best gap in collusive seeds EUR "
          f"{'n/a' if summary['mean_gap_collusive'] is None else round(summary['mean_gap_collusive'])}/round")
    return summary


if __name__ == "__main__":
    for name in ("baseline", "long"):                       # the full learner at both horizons
        if (RESULTS / f"{name}.json").exists():
            main(name, figure=True)
    for control in ("gamma0", "memory0", "gamma0_long", "memory0_long"):   # same tests on the ablations
        if (RESULTS / f"{control}.json").exists():
            main(control, figure=False)
