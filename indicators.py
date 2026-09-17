"""M3-M5 on every trained seed: the driver, the summary JSON and the punishment figure.
The tests themselves are `collusim.indicators`.

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

from collusim import VERDICTS, ablation, punishment, temptation
from experiments import COLORS, FIGURES, HORIZONS, MARKET, RESULTS, load_summary, style

COLLUSIVE = 0.5                                  # a seed "colluded" if its Delta exceeds this


def load_run(name, seed):
    z = np.load(RESULTS / name / f"seed_{seed}.npz")
    return z["Q"], int(z["final_state"])


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
            punishment=[punishment(Q, state, MARKET, deviator=i) for i in range(MARKET.n_firms)],   # [0] = SPEC's agent 0
            temptation=temptation(Q, state, MARKET),
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
