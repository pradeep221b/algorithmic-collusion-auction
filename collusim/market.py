"""Uniform-price electricity auction: clearing rule, benchmarks, payoff tables.

Single node, symmetric firms, inelastic demand, price cap. Everything the learners ever
see comes through `Market.clear`; the payoff tables are that function evaluated once on
every point of the discrete action grid.

    market = Market()                  # the spec: 3 firms x 60 MW, cost 10, demand 100, cap 100, 15 prices
    market = Market(demand=130)        # every firm pivotal: the clearing price is the HIGHEST bid
    market = Market(pay_as_bid=True)   # dispatched firms are paid their own bid
    profit, price = market.tables      # profit[encode(profile, k), i], price[encode(profile, k)]

The module-level names below (PRICES, clear_market, payoff_tables, ...) are the spec market,
kept so the reproduction scripts and the tests read as they always did.
"""
import functools
import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Market:
    """One auction: parameters, clearing rule, benchmarks, payoff tables. Immutable and hashable."""
    n_firms: int = 3
    capacity: float = 60.0      # MW per firm
    cost: float = 10.0          # EUR/MWh, symmetric marginal cost
    demand: float = 100.0       # MW, inelastic
    p_max: float = 100.0        # EUR/MWh price cap
    n_prices: int = 15          # action grid: linspace(cost, p_max, n_prices)
    pay_as_bid: bool = False    # False: everyone dispatched is paid the marginal bid (uniform price)

    @property
    def prices(self):
        return np.linspace(self.cost, self.p_max, self.n_prices)

    @property
    def p_comp(self):
        """Competitive benchmark: the one-shot Nash of the auction is bidding marginal cost."""
        return self.cost

    @property
    def p_coll(self):
        """Collusive benchmark: the joint-profit maximum under inelastic demand is the cap."""
        return self.p_max

    def collusion_index(self, mean_price):
        """Calvano's Delta: 0 = competitive benchmark, 1 = full collusion."""
        return (mean_price - self.p_comp) / (self.p_coll - self.p_comp)

    def clear(self, bids):
        """Clear one round. Returns (clearing_price, dispatch_vector).

        1. Sort bids ascending, dispatch greedily until demand is met.
        2. Clearing price = bid of the last unit dispatched.
        3. Ties at the margin split the residual demand pro rata to capacity.
        """
        bids = np.asarray(bids, dtype=float)
        if bids.min() < 0 or bids.max() > self.p_max:
            raise ValueError(f"bids must lie in [0, {self.p_max}], got {bids}")
        dispatch = np.zeros_like(bids)
        remaining = self.demand
        price = self.p_max  # ponytail: scarcity price if total capacity < demand; never hit with 3x60 vs 100
        for level in np.unique(bids):            # ascending
            tied = bids == level
            available = self.capacity * tied.sum()
            share = min(1.0, remaining / available)   # 1.0 = fully dispatched, <1 = pro-rata split
            dispatch[tied] = self.capacity * share
            remaining -= available * share
            price = level
            if remaining <= 1e-9 * self.demand:   # tolerance: a float residual must not open the next level
                break
        return price, dispatch

    def profits(self, bids):
        """Per-firm profit for one round: (price paid - cost) * quantity dispatched."""
        bids = np.asarray(bids, dtype=float)
        price, dispatch = self.clear(bids)
        return ((bids if self.pay_as_bid else price) - self.cost) * dispatch

    @functools.cached_property
    def tables(self):
        """(profit, price) evaluated once on every action profile; computed on first use, then cached.

        profit[encode(profile, k), i] = firm i's profit when `profile` is played
        price[encode(profile, k)]     = clearing price for `profile`
        15^3 = 3375 profiles x 3 firms for the spec market: the whole game fits in a lookup table.
        """
        n, k = self.n_firms, self.n_prices
        profit = np.zeros((k ** n, n))
        price = np.zeros(k ** n)
        for profile in itertools.product(range(k), repeat=n):
            idx = encode(profile, k)
            bids = self.prices[list(profile)]
            p, dispatch = self.clear(bids)
            profit[idx] = ((bids if self.pay_as_bid else p) - self.cost) * dispatch
            price[idx] = p
        return profit, price


# --- The spec market as module-level names (SPEC.md, "Market model") -----------
DEFAULT = Market()
N_FIRMS, CAPACITY, COST, DEMAND, P_MAX, N_PRICES = (DEFAULT.n_firms, DEFAULT.capacity, DEFAULT.cost,
                                                     DEFAULT.demand, DEFAULT.p_max, DEFAULT.n_prices)
PRICES = DEFAULT.prices          # the action grid: 10.00, 16.43, ... 100.00
P_COMP, P_COLL = DEFAULT.p_comp, DEFAULT.p_coll


def encode(actions, k=N_PRICES):
    """Action profile (a_0, ..., a_{n-1}) -> single integer, base k. Used as the state index.
    Pass k explicitly for any market other than the spec's 15-price grid."""
    idx = 0
    for a in actions:
        idx = idx * k + int(a)
    return idx


def decode(idx, n=N_FIRMS, k=N_PRICES):
    """Inverse of `encode`."""
    out = []
    for _ in range(n):
        out.append(idx % k)
        idx //= k
    return tuple(reversed(out))


def collusion_index(mean_price):
    return DEFAULT.collusion_index(mean_price)


def clear_market(bids, capacity=CAPACITY, demand=DEMAND):
    return Market(capacity=capacity, demand=demand).clear(bids)


def profits(bids):
    return DEFAULT.profits(bids)


def payoff_tables(n=N_FIRMS, k=N_PRICES, pay_as_bid=False):
    return Market(n_firms=n, n_prices=k, pay_as_bid=pay_as_bid).tables
