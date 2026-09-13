"""Phase 1: is the high price learned strategy or frozen learning?

    python phase1.py sweep     # beta sweep, memory 0 vs 1            (~30 min, 50 seeds)
    python phase1.py reinject  # restart converged 5M runs with eps=0.05 (needs results/long, memory0_long)
    python phase1.py residual  # Bellman residual on the temptation action, all configs
    python phase1.py payasbid  # same learners, dispatched firms paid their own bid (mechanism test)
    python phase1.py occupation # constant eps, memory 0: where does the Edgeworth cycle spend its time
    python phase1.py fig       # figures/phase1.png from the json files above

sweep     slower exploration decay -> more undercut attempts before eps dies. If memory-0 Delta
          falls with beta while memory-1 holds, the freeze is an artefact and collusion is not.
reinject  eps0=0.05 then decay again from a converged table. A learned equilibrium re-converges
          to the same price; a stale-Q freeze does not.
payasbid  under uniform price, (low, high, high) is a rest point: the low bidder is paid the high
          price for 60 MW and the high bidders cannot move without an exploration event. Paid
          own bid, the low bidder wants up -> the rest point should vanish for memory-0.
residual  target = r + gamma*max Q[s'] (rivals at their greedy action) minus Q[s, a_dev] for the
          best unplayed deviation. ~0: the table has learned the deviation is bad (continuation).
          >> 0 with target > Q: nobody has tried a_dev since rivals moved -> stale estimate.
Writes results/phase1_<name>.json.
"""
import json
import sys

import numpy as np

import agents as ag
import market as m
import experiments as ex

BETAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
SEEDS = 50


def sweep():
    cfgs = {f"beta{b:g}_mem{mem}": dict(gamma=0.95, memory=mem, beta=b, n_rounds=int(10 / b) + 4_000_000)
            for b in BETAS for mem in (0, 1)}          # horizon: eps reaches e^-10, then a 4M greedy tail
    ex.CONFIGS.update(cfgs)
    out = {}
    for name in cfgs:
        s = ex.run_config(name, range(SEEDS))
        out[name] = dict(beta=cfgs[name]["beta"], memory=cfgs[name]["memory"], **s)
        print(f"{name:16s} Delta={s['mean']:.3f} CI=[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] "
              f"conv={s['frac_converged']:.2f} ({s['seconds']}s)", flush=True)
    (ex.RESULTS / "phase1_sweep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def reinject(beta=ag.BETA, n_rounds=2_000_000):
    out = {}
    for src in ("long", "memory0_long"):
        cfg = ex.CONFIGS[src]
        for eps0 in (0.05, 0.5):
            before, after, min_price, swapped = [], [], [], []
            for r in ex.load_summary(src)["seeds"]:
                Q = np.load(ex.RESULTS / src / f"seed_{r['seed']}.npz")["Q"]
                low0 = int(np.argmin([np.argmax(Q[i, 0]) for i in range(3)]))   # who is the low bidder (memory 0)
                _, _, s, bsum, bcnt = ag.train(Q, ex.PROFIT, ex.PRICE, ag.ALPHA, cfg["gamma"], beta, n_rounds,
                                               ag.CONV_ROUNDS, ag.BLOCK, 10_000 + r["seed"], eps0)
                play = ag.greedy_play(Q, ex.PROFIT, ex.PRICE, s, 1_000)
                before.append(r["delta"])
                after.append(float(m.collusion_index(play["prices"][100:].mean())))
                min_price.append(float((bsum / np.maximum(bcnt, 1)).min()))    # how far the perturbation pushed prices
                swapped.append(int(np.argmin([np.argmax(Q[i, 0]) for i in range(3)])) != low0)
            d = np.array(after) - np.array(before)
            key = f"{src}_eps{eps0:g}"
            out[key] = dict(eps0=eps0, before=ex.summarise(before), after=ex.summarise(after),
                            change_mean=float(d.mean()), frac_dropped_0_1=float((d < -0.1).mean()),
                            min_block_price_mean=float(np.mean(min_price)),
                            frac_low_bidder_swapped=float(np.mean(swapped)) if cfg["memory"] == 0 else None)
            print(f"{key:20s} Delta {out[key]['before']['mean']:.3f} -> {out[key]['after']['mean']:.3f}  "
                  f"dropped>0.1: {out[key]['frac_dropped_0_1']:.2f}  min price {out[key]['min_block_price_mean']:.0f}  "
                  f"role swap: {out[key]['frac_low_bidder_swapped']}", flush=True)
    (ex.RESULTS / "phase1_reinject.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def payasbid():
    ex.PROFIT, ex.PRICE = m.payoff_tables(pay_as_bid=True)
    cfgs = {"pab_mem0": dict(gamma=0.95, memory=0, n_rounds=5_000_000),
            "pab_mem1": dict(gamma=0.95, memory=1, n_rounds=5_000_000)}
    ex.CONFIGS.update(cfgs)
    out = {}
    for name in cfgs:
        s = ex.run_config(name, range(SEEDS))
        out[name] = dict(**cfgs[name], **s)
        print(f"{name:10s} Delta={s['mean']:.3f} CI=[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] "
              f"conv={s['frac_converged']:.2f} ({s['seconds']}s)", flush=True)
    (ex.RESULTS / "phase1_payasbid.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def occupation(n_rounds=3_000_000, seeds=20, memory=0, epss=(0.001, 0.003, 0.01, 0.03, 0.1), tag=None):
    """Constant eps (beta=0), memory 0. If the time-average Delta over the last 2M rounds matches the
    frozen Delta (0.95), the freeze is simply the cycle's occupation measure: falls are slow chains of
    learning events, rises are single free jumps, so the process sits near the top most of the time."""
    out = {}
    for eps in epss:
        deltas, frac_high, falls, rises = [], [], [], []
        for seed in range(seeds):
            Q = ag.initial_q(ex.PROFIT, 0.95, memory)
            _, _, _, bsum, bcnt = ag.train(Q, ex.PROFIT, ex.PRICE, ag.ALPHA, 0.95, 0.0, n_rounds,
                                           10**9, ag.BLOCK, 20_000 + seed, eps)
            p = (bsum / np.maximum(bcnt, 1))[len(bsum) // 3:]   # drop the first third as burn-in
            deltas.append(float(m.collusion_index(p.mean())))
            frac_high.append(float((p > 55).mean()))
            d = np.diff(p)
            falls.append(float(-d[d < -5].sum() / max((d < -5).sum(), 1)))   # mean size of a block-to-block fall
            rises.append(float(d[d > 5].sum() / max((d > 5).sum(), 1)))
        out[f"eps{eps:g}"] = dict(eps=eps, memory=memory, delta=ex.summarise(deltas), frac_blocks_price_above_55=float(np.mean(frac_high)),
                                  mean_fall=float(np.mean(falls)), mean_rise=float(np.mean(rises)))
        print(f"eps={eps:<6g} time-avg Delta={np.mean(deltas):.3f}  blocks>55: {np.mean(frac_high):.2f}  "
              f"mean fall {np.mean(falls):.0f} / mean rise {np.mean(rises):.0f}", flush=True)
    tag = tag if tag is not None else ("" if memory == 0 else "_mem1")
    (ex.RESULTS / f"phase1_occupation{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def occupation_mem1():
    occupation(memory=1)


def occupation_mem1_long():
    """Does the memory-1 curve ever meet its frozen 0.62? 30M rounds at eps=1e-4, 20 seeds."""
    occupation(n_rounds=30_000_000, memory=1, epss=(1e-4,), tag="_mem1_long")


def bellman_residual(Q, state, gamma, rounds=100):
    """Per agent, mean over cycle states of target(a_dev) - Q[s, a_dev] and the same for the played action."""
    play = ag.greedy_play(Q, ex.PROFIT, ex.PRICE, state, rounds)
    n, S, k = Q.shape
    res_dev, res_played, dev_better = [], [], []
    for t, a in enumerate(play["actions"]):
        s = play["states"][t]
        for i in range(n):
            alts = np.array([ex.PROFIT[m.encode([*a[:i], b, *a[i + 1:]]), i] for b in range(k)])
            b = int(np.flatnonzero(alts == alts.max()).max())
            if alts[b] - alts[a[i]] <= 1e-9:
                continue                                   # no temptation here
            for act, store in ((b, res_dev), (a[i], res_played)):
                profile = m.encode([*a[:i], act, *a[i + 1:]])
                nxt = profile if S > 1 else 0
                target = ex.PROFIT[profile, i] + gamma * Q[i, nxt].max()
                store.append(target - Q[i, s, act])
            # does a one-step backup already say the deviation beats the played action?
            tgt_dev = res_dev[-1] + Q[i, s, b]
            dev_better.append(tgt_dev > Q[i, s, a[i]])
    f = lambda x: float(np.mean(x)) if x else float("nan")
    return dict(residual_dev=f(res_dev), residual_played=f(res_played),
                frac_backup_prefers_dev=f(dev_better), n_temptations=len(res_dev))


def residual():
    out = {}
    for name in ("long", "gamma0_long", "memory0_long"):
        if not (ex.RESULTS / f"{name}.json").exists():
            continue
        gamma = ex.CONFIGS[name]["gamma"]
        rows = []
        for r in ex.load_summary(name)["seeds"]:
            z = np.load(ex.RESULTS / name / f"seed_{r['seed']}.npz")
            rows.append(dict(seed=r["seed"], delta=r["delta"], **bellman_residual(z["Q"], int(z["final_state"]), gamma)))
        dev = np.array([x["residual_dev"] for x in rows])
        out[name] = dict(seeds=rows, residual_dev=ex.summarise(dev[~np.isnan(dev)]),
                         residual_played_mean=float(np.nanmean([x["residual_played"] for x in rows])),
                         frac_backup_prefers_dev=float(np.nanmean([x["frac_backup_prefers_dev"] for x in rows])))
        print(f"{name:13s} residual(dev) {out[name]['residual_dev']['mean']:8.0f}  residual(played) "
              f"{out[name]['residual_played_mean']:8.0f}  backup prefers dev: {out[name]['frac_backup_prefers_dev']:.2f}",
              flush=True)
    (ex.RESULTS / "phase1_residual.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = ex.COLORS
    load = lambda n: json.loads((ex.RESULTS / f"phase1_{n}.json").read_text(encoding="utf-8"))
    sw, oc, rs = load("sweep"), load("occupation"), load("residual")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)

    for mem, col, lab in ((0, C["s2"], "memory 0 (no past)"), (1, C["s1"], "memory 1 (spec)")):
        rows = sorted((v for v in sw.values() if v["memory"] == mem), key=lambda v: v["beta"])
        b = [v["beta"] for v in rows]
        a1.errorbar(b, [v["mean"] for v in rows],
                    yerr=[[v["mean"] - v["ci95"][0] for v in rows], [v["ci95"][1] - v["mean"] for v in rows]],
                    color=col, marker="o", capsize=3, label=lab)
    a1.set_xscale("log"); a1.set_xlabel("exploration decay β  (ε_t = e^{-βt})", color=C["secondary"])
    a1.set_ylabel("Δ after freeze", color=C["secondary"]); a1.set_ylim(0, 1.05)
    a1.axvline(ag.BETA, color=C["muted"], linestyle="--", linewidth=1); a1.text(ag.BETA, 0.03, " spec", fontsize=8, color=C["secondary"])
    a1.set_title("A  slower exploration decay does not dissolve the freeze", loc="left", fontsize=10, color=C["ink"])
    a1.legend(frameon=False, fontsize=8, loc="lower left")

    oc1 = load("occupation_mem1") if (ex.RESULTS / "phase1_occupation_mem1.json").exists() else {}
    if (ex.RESULTS / "phase1_occupation_mem1_long.json").exists():
        oc1.update(load("occupation_mem1_long"))
    for data, col, mem in ((oc, C["s2"], 0), (oc1, C["s1"], 1)):
        if not data:
            continue
        rows = sorted(data.values(), key=lambda v: v["eps"])
        e = [v["eps"] for v in rows]
        a2.errorbar(e, [v["delta"]["mean"] for v in rows],
                    yerr=[[v["delta"]["mean"] - v["delta"]["ci95"][0] for v in rows], [v["delta"]["ci95"][1] - v["delta"]["mean"] for v in rows]],
                    color=col, marker="o", capsize=3, label=f"memory {mem}, constant ε, time average")
        frozen = sw[f"beta1e-05_mem{mem}"]["mean"]
        a2.axhline(frozen, color=col, linestyle="--", linewidth=1, label=f"memory {mem}, frozen Δ = {frozen:.2f}")
    a2.set_xscale("log"); a2.set_xlabel("constant exploration rate ε", color=C["secondary"])
    a2.set_ylabel("Δ, time average over 2M rounds", color=C["secondary"]); a2.set_ylim(0, 1.05)
    a2.set_title("B  both freeze where they live; memory 1 needs 30× less noise", loc="left", fontsize=10, color=C["ink"])
    a2.legend(frameon=False, fontsize=8, loc="lower left")

    names = [n for n in ("long", "gamma0_long", "memory0_long") if n in rs]
    labels = {"long": "full learner\nγ=0.95, memory 1", "gamma0_long": "γ = 0\n(no future)", "memory0_long": "memory 0\n(no past)"}
    rng = np.random.default_rng(0)
    for x, n in enumerate(names):
        f = np.array([r["frac_backup_prefers_dev"] for r in rs[n]["seeds"]])
        f = f[~np.isnan(f)]
        a3.bar(x, f.mean(), width=0.5, color=C["s1"], alpha=0.85, zorder=2)
        a3.scatter(x + rng.uniform(-0.18, 0.18, len(f)), f, s=9, color=C["secondary"], alpha=0.5, zorder=4)
    a3.set_xticks(range(len(names)), [labels[n] for n in names])
    a3.set_ylabel("share of temptations where a one-step Bellman backup\nalready prefers the unplayed deviation", color=C["secondary"], fontsize=8)
    a3.set_ylim(0, 1.05)
    a3.set_title("C  stale estimate (→1) vs learned continuation (→0)", loc="left", fontsize=10, color=C["ink"])
    for a in (a1, a2, a3):
        ex.style(a)
    fig.tight_layout()
    ex.FIGURES.mkdir(exist_ok=True)
    fig.savefig(ex.FIGURES / "phase1.png")
    plt.close(fig)


if __name__ == "__main__":
    {"sweep": sweep, "reinject": reinject, "residual": residual, "payasbid": payasbid, "occupation": occupation, "occupation_mem1": occupation_mem1, "occupation_mem1_long": occupation_mem1_long, "fig": fig}[sys.argv[1]]()
