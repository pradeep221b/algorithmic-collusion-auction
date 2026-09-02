"""M2 seed sweep and M4 ablations.

    python experiments.py            # 50 seeds x 4 configurations, ~4 minutes on a laptop
    python experiments.py --seeds 20 # the spec's minimum

Writes results/<config>.json (per-seed Delta, convergence), results/<config>/seed_k.npz
(Q-tables, needed by indicators.py) and figures/price_trajectory.png, figures/ablation_bars.png.
"""
import argparse
import json
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import agents as ag
import market as m

RESULTS = pathlib.Path(__file__).parent / "results"
FIGURES = pathlib.Path(__file__).parent / "figures"
PROFIT, PRICE = m.payoff_tables()

CONFIGS = {
    "baseline":     dict(gamma=0.95, memory=1, n_rounds=1_000_000),   # SPEC M2, as specified
    "gamma0":       dict(gamma=0.0,  memory=1, n_rounds=1_000_000),   # M4: agents do not value the future
    "memory0":      dict(gamma=0.95, memory=0, n_rounds=1_000_000),   # M4: agents do not remember the past
    "long":         dict(gamma=0.95, memory=1, n_rounds=5_000_000),   # the 1e6 horizon binds: same, converged
    "gamma0_long":  dict(gamma=0.0,  memory=1, n_rounds=5_000_000),   # M4 at the converged horizon
    "memory0_long": dict(gamma=0.95, memory=0, n_rounds=5_000_000),
}
HORIZONS = {"1M": ("baseline", "gamma0", "memory0"), "5M": ("long", "gamma0_long", "memory0_long")}

# Chart chrome (dataviz reference palette): one ink, one muted, three categorical slots.
COLORS = dict(ink="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9",
              s1="#2a78d6", s2="#eb6834", s3="#1baf7a")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["secondary"], labelsize=9)
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def run_seed(seed, gamma, memory, n_rounds, alpha=ag.ALPHA, beta=ag.BETA):
    """Train one seed to the horizon, then measure Delta on frozen greedy play."""
    Q = ag.initial_q(PROFIT, gamma, memory)
    t_conv, streak, s, bsum, bcnt = ag.train(Q, PROFIT, PRICE, alpha, gamma, beta, n_rounds,
                                             ag.CONV_ROUNDS, ag.BLOCK, seed)
    trajectory = np.where(bcnt > 0, bsum / np.maximum(bcnt, 1), np.nan)
    # Delta is measured on what the learned policies DO, not on the noisy training trace:
    # play greedy from the final state, drop a short transient, average the clearing price.
    play = ag.greedy_play(Q, PROFIT, PRICE, s, 1_000)
    mean_price = float(play["prices"][100:].mean())
    return dict(seed=seed, Q=Q, t_conv=int(t_conv), converged=bool(streak >= ag.CONV_ROUNDS), trajectory=trajectory,
                final_state=int(play["final_state"]), mean_price=mean_price,
                delta=float(m.collusion_index(mean_price)),
                last_profile=[int(x) for x in play["actions"][-1]])


def summarise(deltas):
    d = np.asarray(deltas, dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else None
    ci = [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)] if se is not None else [None, None]
    return dict(n=int(len(d)), mean=float(d.mean()), ci95=ci,
                median=float(np.median(d)), min=float(d.min()), max=float(d.max()),
                frac_above_0_5=float((d > 0.5).mean()))


def run_config(name, seeds):
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    for seed in seeds:
        r = run_seed(seed, **CONFIGS[name])
        np.savez_compressed(out / f"seed_{seed}.npz", Q=r["Q"], final_state=r["final_state"],
                            trajectory=r["trajectory"], t_conv=r["t_conv"])
        rows.append({k: r[k] for k in ("seed", "delta", "mean_price", "t_conv", "converged", "last_profile")})
    summary = summarise([r["delta"] for r in rows])
    summary["frac_converged"] = float(np.mean([r["converged"] for r in rows]))
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    (RESULTS / f"{name}.json").write_text(json.dumps(
        {"config": CONFIGS[name], "summary": summary, "seeds": rows}, indent=1), encoding="utf-8")
    return summary


def load_summary(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


# --- Figures ---------------------------------------------------------------------
def fig_price_trajectory(name="baseline"):
    data = load_summary(name)
    trajs = [np.load(RESULTS / name / f"seed_{r['seed']}.npz")["trajectory"] for r in data["seeds"]]
    T = np.arange(len(trajs[0])) * ag.BLOCK / 1e3
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for tr in trajs:
        ax.plot(T, tr, color=COLORS["s1"], linewidth=0.6, alpha=0.18)
    ax.plot(T, np.nanmean(trajs, axis=0), color=COLORS["ink"], linewidth=2, label=f"mean of {len(trajs)} seeds")
    ax.plot([], [], color=COLORS["s1"], linewidth=0.8, alpha=0.6, label="individual seeds")
    for y, txt in ((m.P_COLL, "collusive  €100"), (m.P_COMP, "competitive  €10")):
        ax.axhline(y, color=COLORS["muted"], linewidth=1, linestyle="--")
        ax.text(T[-1], y, txt, ha="right", va="bottom", fontsize=8, color=COLORS["secondary"])
    ax.set_xlabel("round (thousands)", color=COLORS["secondary"])
    ax.set_ylabel("mean clearing price per 1,000 rounds, €/MWh", color=COLORS["secondary"])
    ax.set_ylim(0, 108)
    ax.set_title(f"M2  price trajectory  ({name}: γ={data['config']['gamma']}, memory={data['config']['memory']}, "
                 f"{data['config']['n_rounds'] // 10**6}M rounds)", loc="left", fontsize=11, color=COLORS["ink"])
    ax.legend(frameon=False, fontsize=8, loc="center right")
    style(ax)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / ("price_trajectory.png" if name == "baseline" else f"price_trajectory_{name}.png"))
    plt.close(fig)


def fig_ablation_bars():
    """Delta per configuration, grouped by horizon: full learner vs. no-future vs. no-past."""
    labels = ["full learner\nγ=0.95, memory 1", "γ = 0\n(no future)", "memory 0\n(no past)"]
    groups = [(h, names) for h, names in HORIZONS.items() if all((RESULTS / f"{n}.json").exists() for n in names)]
    if not groups:          # partial run (e.g. --configs baseline): nothing to compare yet
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.4), dpi=150, sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (horizon, names) in zip(axes[0], groups):
        for x, name in enumerate(names):
            data = load_summary(name)
            s, d = data["summary"], np.array([r["delta"] for r in data["seeds"]])
            ax.bar(x, s["mean"], width=0.5, color=COLORS["s1"], alpha=0.85, zorder=2)
            ax.errorbar(x, s["mean"], yerr=[[s["mean"] - s["ci95"][0]], [s["ci95"][1] - s["mean"]]],
                        color=COLORS["ink"], capsize=4, linewidth=1.2, zorder=3)
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(d)), d, s=9, color=COLORS["secondary"], alpha=0.5, zorder=4)
            ax.text(x, min(d.max(), 0.98) + 0.03, f"Δ = {s['mean']:.2f}", ha="center", fontsize=9, color=COLORS["ink"])
        ax.set_xticks(range(len(names)), labels)
        ax.set_title(f"M4  ablation, {horizon} rounds", loc="left", fontsize=11, color=COLORS["ink"])
        style(ax)
    axes[0][0].set_ylim(0, 1.08)
    axes[0][0].set_ylabel("collusion index Δ  (bar = mean, whisker = 95% CI, dots = seeds)",
                          color=COLORS["secondary"], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_bars.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50, help="seeds per configuration (SPEC minimum 20)")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = p.parse_args()
    for name in args.configs:
        s = run_config(name, range(args.seeds))
        print(f"{name:9s} n={s['n']} Delta={s['mean']:.3f} CI95=[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] "
              f"median={s['median']:.3f} frac(Delta>0.5)={s['frac_above_0_5']:.2f} "
              f"converged={s['frac_converged']:.2f}  ({s['seconds']}s)", flush=True)
    for name in ("baseline", "long"):
        if (RESULTS / f"{name}.json").exists():
            fig_price_trajectory(name)
    fig_ablation_bars()


if __name__ == "__main__":
    main()
