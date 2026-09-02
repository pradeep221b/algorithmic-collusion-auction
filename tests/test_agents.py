"""Q-learning mechanics: initialisation, update rule, state encoding, greedy play, determinism."""
import numpy as np
import pytest

import agents as ag
import market as m

PROFIT, PRICE = m.payoff_tables()


def test_initial_q_is_discounted_expected_profit():
    Q = ag.initial_q(PROFIT, gamma=0.95, memory=1)
    assert Q.shape == (3, 3375, 15)
    # every state row identical (init carries no history), rows finite and non-negative
    assert np.allclose(Q[0, 0], Q[0, 1234])
    assert np.all(Q >= 0) and np.all(np.isfinite(Q))
    # hand computation, mean over rivals of own profit / (1-0.95), for every agent position
    # (agent 0 is the identity case of the axis shuffle; agents 1 and 2 exercise it)
    for i, own in ((0, 14), (1, 7), (2, 3)):
        idxs = [m.encode([*(r[:i]), own, *(r[i:])]) for r in ((j, l) for j in range(15) for l in range(15))]
        assert Q[i, 0, own] == pytest.approx(PROFIT[idxs, i].mean() / 0.05)


def test_memory0_has_single_state():
    Q = ag.initial_q(PROFIT, gamma=0.95, memory=0)
    assert Q.shape == (3, 1, 15)


def test_train_update_matches_formula_for_one_round():
    # alpha=1, gamma=0, beta huge -> first round is fully random, update is Q[s,a] = r exactly.
    # Sentinel -1 so a zero-profit write is still visible as a change.
    Q = np.full((3, 3375, 15), -1.0)
    t_conv, _, s_final, bs, bc = ag.train(Q, PROFIT, PRICE, 1.0, 0.0, 1e9, 1, 10, 1, seed=3)
    changed = np.argwhere(Q != -1.0)
    assert len(changed) == 3                          # one cell per agent
    assert len({s for _, s, _ in changed}) == 1       # all three at the same (initial) state
    for i, s, a in changed:
        assert Q[i, s, a] == pytest.approx(PROFIT[s_final, i])
        assert m.decode(s_final)[i] == a              # the played action is the column written
    assert bc.sum() == 1 and bs[0] == PRICE[s_final]


def test_train_bootstraps_from_the_next_state():
    # alpha=1, gamma=0.5, one random round: Q[s,a] = r + 0.5 * max_a' Q0[s', a'] with s' = the
    # profile just played. A kernel that bootstrapped from the current state instead would fail.
    rng = np.random.default_rng(11)
    Q0 = rng.uniform(0, 1000, size=(3, 3375, 15))
    Q = Q0.copy()
    _, _, s_final, _, _ = ag.train(Q, PROFIT, PRICE, 1.0, 0.5, 1e9, 1, 10, 1, seed=4)
    changed = np.argwhere(Q != Q0)
    assert len(changed) == 3
    for i, s, a in changed:
        assert Q[i, s, a] == pytest.approx(PROFIT[s_final, i] + 0.5 * Q0[i, s_final].max())


def test_s_next_is_the_profile_just_played():
    # The bug the spec warns about: s' must be the profile played this round, not the one before.
    # alpha=1, gamma=0, beta=0 -> two fully random rounds, each writing Q[i, s, a_i] = r exactly.
    # Sentinel -1 so a legitimately-zero reward is distinguishable from an unwritten cell.
    Q = np.full((3, 3375, 15), -1.0)
    _, _, final, _, _ = ag.train(Q, PROFIT, PRICE, 1.0, 0.0, 0.0, 2, 10**9, 10, seed=5)
    a2 = m.decode(final)                       # round-2 profile == final state
    # round 2 wrote Q[i, s1, a2_i] = profit(final, i) at one shared state s1 for all agents
    s1_set = set.intersection(*[set(np.nonzero(Q[i, :, a2[i]] == PROFIT[final, i])[0]) for i in range(3)])
    assert s1_set
    # and s1 must itself be round 1's profile: round 1 wrote Q[i, s0, a1_i] = profit(s1, i)
    found = False
    for s1 in s1_set:
        a1 = m.decode(s1)
        s0_set = set.intersection(*[set(np.nonzero(Q[i, :, a1[i]] == PROFIT[s1, i])[0]) - {s1}
                                    for i in range(3)])
        found |= bool(s0_set)
    assert found


def test_train_is_deterministic_per_seed():
    Q1 = ag.initial_q(PROFIT, 0.95, 1)
    Q2 = ag.initial_q(PROFIT, 0.95, 1)
    r1 = ag.train(Q1, PROFIT, PRICE, 0.15, 0.95, 1e-5, 20_000, 10**9, 1000, seed=7)
    r2 = ag.train(Q2, PROFIT, PRICE, 0.15, 0.95, 1e-5, 20_000, 10**9, 1000, seed=7)
    assert np.array_equal(Q1, Q2) and r1[2] == r2[2]
    Q3 = ag.initial_q(PROFIT, 0.95, 1)
    ag.train(Q3, PROFIT, PRICE, 0.15, 0.95, 1e-5, 20_000, 10**9, 1000, seed=8)
    assert not np.array_equal(Q1, Q3)


def test_convergence_fires_when_policy_is_stable():
    # alpha=0: Q never changes, so the greedy policy is stable from round 0.
    Q = ag.initial_q(PROFIT, 0.95, 1)
    t_conv, *_ = ag.train(Q, PROFIT, PRICE, 0.0, 0.95, 1e-5, 5_000, 1_000, 100, seed=0)
    assert t_conv == 999


def test_greedy_play_follows_argmax_and_honours_force():
    Q = np.zeros((3, 3375, 15))
    Q[:, :, 14] = 1.0                         # everyone's greedy action is the cap
    out = ag.greedy_play(Q, PROFIT, PRICE, state=0, n_rounds=3, force={1: {0: 0}})
    assert out["actions"].tolist() == [[14, 14, 14], [0, 14, 14], [14, 14, 14]]
    assert out["prices"].tolist() == [100, 100, 100]     # price = 2nd-lowest bid, unchanged
    assert out["profits"][1].tolist() == pytest.approx([90 * 60, 90 * 20, 90 * 20])
    assert out["states"][2] == m.encode((0, 14, 14))     # state = last profile
