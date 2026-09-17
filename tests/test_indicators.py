"""The three indicators, checked on hand-built policies whose behaviour is known exactly.

A synthetic Q-table whose greedy policy we choose by hand lets us verify the indicator
code independently of whether learning ever produces collusion.
"""
import numpy as np
import pytest

from collusim import indicators as ind, market as m

CAP, FLOOR = m.N_PRICES - 1, 0
ALL_CAP = m.encode((CAP,) * 3)
ALL_FLOOR = m.encode((FLOOR,) * 3)


def policy(rule):
    """Q whose greedy action in state s is rule(profile) for every agent."""
    Q = np.zeros((3, m.N_PRICES ** 3, m.N_PRICES))
    for s in range(Q.shape[1]):
        Q[:, s, rule(m.decode(s))] = 1.0
    return Q


def policy_per_agent(rules):
    """Q whose greedy action for agent i in state s is rules[i](profile)."""
    Q = np.zeros((3, m.N_PRICES ** 3, m.N_PRICES))
    for s in range(Q.shape[1]):
        for i, rule in enumerate(rules):
            Q[i, s, rule(m.decode(s))] = 1.0
    return Q


tit_for_tat = policy(lambda p: CAP if p in ((CAP,) * 3, (FLOOR,) * 3) else FLOOR)   # punish once, then forgive
grim = policy(lambda p: CAP if p == (CAP,) * 3 else FLOOR)                            # punish forever
always_cap = policy(lambda p: CAP)                                                     # ignores rivals
always_floor = policy(lambda p: FLOOR)                                                 # competitive


def test_punishment_detects_punish_then_forgive():
    r = ind.punishment(tit_for_tat, ALL_CAP, m.DEFAULT)
    assert r["verdict"] == "punish_forgive"
    p = r["prices"]
    d = r["rounds"].index(0)
    assert p[d] == 100          # deviation round: price = 2nd-lowest bid, still the cap
    assert p[d + 1] == 10       # rivals punish: everyone at the floor
    assert p[d + 2] == 100      # ...and forgive
    assert r["profits"][d][0] == pytest.approx(90 * 60)   # the deviator's one-round gain
    assert r["rival_drop_early"] == pytest.approx(90 / 12)  # one round of 90 lower, averaged over 12


def test_punishment_detects_grim_trigger():
    r = ind.punishment(grim, ALL_CAP, m.DEFAULT)
    assert r["verdict"] == "punish_no_recovery"
    assert r["p_late"] == 10


def test_punishment_detects_no_reaction():
    r = ind.punishment(always_cap, ALL_CAP, m.DEFAULT)
    assert r["verdict"] == "no_reaction"
    assert not r["deviation_was_noop"]


def test_punishment_flags_noop_deviation():
    r = ind.punishment(always_floor, ALL_FLOOR, m.DEFAULT)
    assert r["verdict"] == "no_reaction" and r["deviation_was_noop"]


def test_punishment_ignores_self_inflicted_price_drop():
    # Rivals never move (agent 1 always bids EUR 55, agent 2 the cap). Agent 0 bids the cap
    # unless its own last bid was the floor, in which case it stays at the floor. After the
    # forced undercut the PRICE falls from 100 to 55 for good, but no rival reacted.
    sticky = lambda p: FLOOR if p[0] == FLOOR else CAP
    Q = policy_per_agent([sticky, lambda p: 7, lambda p: CAP])
    r = ind.punishment(Q, m.encode((CAP, 7, CAP)), m.DEFAULT)
    assert r["p_pre"] == 100 and r["p_late"] == pytest.approx(m.PRICES[7])
    assert r["verdict"] == "no_reaction"


def test_temptation_at_collusive_profile():
    r = ind.temptation(always_cap, ALL_CAP, m.DEFAULT)
    assert r["verdict"] == "unclaimed_temptation"
    # each agent earns 90 * 100/3 = 3000; any undercut earns 90 * 60 = 5400 at the same price
    assert r["played_profit"] == pytest.approx([3000] * 3)
    assert r["gap_max"] == pytest.approx([2400] * 3)
    assert r["best_deviation"] == [CAP - 1] * 3     # ties broken toward the smallest undercut


def test_temptation_at_competitive_profile_is_static_nash():
    r = ind.temptation(always_floor, ALL_FLOOR, m.DEFAULT)
    assert r["verdict"] == "static_nash"
    assert r["gap_max"] == pytest.approx([0, 0, 0])
    assert r["frac_rounds_static_nash"] == 1.0


def test_ablation_verdict():
    base = {"mean": 0.7}
    assert ind.ablation(base, {"mean": 0.1}, {"mean": 0.2})["verdict"] == "pass"
    assert ind.ablation(base, {"mean": 0.6}, {"mean": 0.1})["verdict"] == "fail"
    assert ind.ablation({"mean": 0.3}, {"mean": 0.2}, {"mean": 0.2})["verdict"] == "fail"  # not < half
