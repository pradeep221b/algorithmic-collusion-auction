"""M1 acceptance tests: clearing rule, pro-rata split, benchmarks."""
import numpy as np
import pytest

from collusim import market as m


def test_all_equal_bids_split_pro_rata():
    price, dispatch = m.clear_market([50, 50, 50])
    assert price == 50
    assert np.allclose(dispatch, [100 / 3] * 3)
    assert abs(dispatch.sum() - m.DEMAND) < 1e-9


def test_strict_undercut_sets_price_at_marginal_unit():
    price, dispatch = m.clear_market([10, 50, 100])
    assert price == 50                      # the 40 MW residual comes from the 50-bidder
    assert np.allclose(dispatch, [60, 40, 0])   # highest bidder is excluded entirely


def test_tie_at_the_margin_splits_residual():
    price, dispatch = m.clear_market([10, 50, 50])
    assert price == 50
    assert np.allclose(dispatch, [60, 20, 20])  # 40 MW residual split pro rata


def test_bid_above_cap_is_rejected():
    with pytest.raises(ValueError):
        m.clear_market([10, 50, 101])


def test_dispatch_sums_to_demand_for_random_bids():
    rng = np.random.default_rng(0)
    for _ in range(200):
        bids = rng.choice(m.PRICES, size=m.N_FIRMS)
        _, dispatch = m.clear_market(bids)
        assert abs(dispatch.sum() - m.DEMAND) < 1e-9


def test_benchmarks_and_index():
    assert m.P_COMP == 10 and m.P_COLL == 100
    assert m.collusion_index(10) == 0
    assert m.collusion_index(100) == 1
    assert m.collusion_index(55) == pytest.approx(0.5)


def test_competitive_price_is_a_one_shot_nash_equilibrium():
    # At all-marginal-cost, no unilateral move earns anything: raise -> excluded, price stays 10.
    base = m.profits([10, 10, 10])
    assert np.allclose(base, 0)
    for p in m.PRICES:
        assert m.profits([p, 10, 10])[0] == pytest.approx(0)


def test_symmetric_high_price_invites_undercutting():
    # Any symmetric price above cost is NOT a one-shot equilibrium: one grid step down
    # captures 60 MW at the unchanged clearing price instead of a 1/3 share.
    for j in range(1, m.N_PRICES):
        p, p_below = m.PRICES[j], m.PRICES[j - 1]
        stay = m.profits([p, p, p])[0]
        undercut = m.profits([p_below, p, p])[0]
        assert undercut > stay


def test_encode_decode_roundtrip():
    for idx in range(m.N_PRICES ** m.N_FIRMS):
        assert m.encode(m.decode(idx)) == idx


def test_payoff_tables_match_clearing():
    profit, price = m.payoff_tables()
    assert profit.shape == (3375, 3) and price.shape == (3375,)
    idx = m.encode((0, 7, 14))
    assert price[idx] == m.PRICES[7]
    assert np.allclose(profit[idx], m.profits(m.PRICES[[0, 7, 14]]))


def test_pay_as_bid_pays_own_bid():
    profit, price = m.payoff_tables(pay_as_bid=True)
    a = m.encode([0, 14, 14])                      # bids 10, 100, 100: low firm sells 60 MW at its own 10
    assert profit[a, 0] == 0 and price[a] == 100
    assert profit[a, 1] == profit[a, 2] == (100 - 10) * 20
