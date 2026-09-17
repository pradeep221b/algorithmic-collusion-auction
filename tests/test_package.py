"""The package surface: a non-default market, the environment, one-call training."""
import pytest

import collusim as cs


def test_pivotal_market_clears_at_highest_bid():
    price, dispatch = cs.Market(demand=130).clear([10, 50, 100])   # all three firms needed
    assert price == 100 and list(dispatch) == [60, 60, 10]


def test_tables_cached_and_market_hashable():
    mk = cs.Market()
    assert mk.tables[0] is mk.tables[0]
    assert {mk: 1}[cs.Market()] == 1


def test_env_rewards_match_tables():
    env = cs.AuctionEnv(cs.Market(), memory=1)
    s0 = env.reset(seed=1)
    assert 0 <= s0 < 15 ** 3
    obs, rewards, terminated, truncated, info = env.step([14, 14, 14])
    assert obs == cs.encode([14, 14, 14], 15) and not terminated
    assert rewards == pytest.approx([90 * 100 / 3] * 3) and info["price"] == 100
    assert cs.AuctionEnv(cs.Market(), memory=0).reset(seed=1) == 0


def test_run_seed_on_a_two_firm_five_price_market():
    mk = cs.Market(n_firms=2, n_prices=5)
    r = cs.run_seed(mk, seed=0, gamma=0.95, memory=1, n_rounds=20_000)
    assert r["Q"].shape == (2, 25, 5) and 0 <= r["delta"] <= 1 and len(r["last_profile"]) == 2
    assert r["delta"] == cs.run_seed(mk, seed=0, gamma=0.95, memory=1, n_rounds=20_000)["delta"]   # deterministic
