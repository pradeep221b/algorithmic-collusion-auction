"""Uniform-price electricity auction: clearing rule, benchmarks, payoff tables.

Single node, three symmetric firms, inelastic demand, price cap. Everything the
learners ever see comes through `clear_market`; the payoff tables are just that
function evaluated once on every point of the discrete action grid.
"""
import itertools

import numpy as np

# --- Parameters (SPEC.md, "Market model") --------------------------------------
N_FIRMS = 3
CAPACITY = 60.0     # MW per firm; total 180 > D, any two firms cover D -> nobody pivotal
COST = 10.0         # EUR/MWh, symmetric marginal cost
DEMAND = 100.0      # MW, inelastic
P_MAX = 100.0       # EUR/MWh price cap
N_PRICES = 15
PRICES = np.linspace(COST, P_MAX, N_PRICES)   # the action grid: 10.00, 16.43, ... 100.00

# --- Benchmarks ----------------------------------------------------------------
P_COMP = COST   # one-shot Nash: Bertrand undercutting drives price to marginal cost
P_COLL = P_MAX  # joint-profit maximum under inelastic demand is the cap


def collusion_index(mean_price):
    """Calvano's Delta: 0 = competitive benchmark, 1 = full collusion."""
    return (mean_price - P_COMP) / (P_COLL - P_COMP)


# --- Clearing ------------------------------------------------------------------
def clear_market(bids, capacity=CAPACITY, demand=DEMAND):
    """Clear one round. Returns (clearing_price, dispatch_vector).

    1. Sort bids ascending, dispatch greedily until demand is met.
    2. Clearing price = bid of the last unit dispatched; everyone dispatched is paid it.
    3. Ties at the margin split the residual demand pro rata to capacity.
    """
    bids = np.asarray(bids, dtype=float)
    if bids.min() < 0 or bids.max() > P_MAX:
        raise ValueError(f"bids must lie in [0, {P_MAX}], got {bids}")
    cap = np.full(bids.shape, capacity, dtype=float)
    dispatch = np.zeros_like(bids)
    remaining = demand
    price = P_MAX  # ponytail: scarcity price if total capacity < demand; never hit with 3x60 vs 100
    for level in np.unique(bids):            # ascending
        tied = bids == level
        available = cap[tied].sum()
        share = min(1.0, remaining / available)   # 1.0 = fully dispatched, <1 = pro-rata split
        dispatch[tied] = cap[tied] * share
        remaining -= available * share
        price = level
        if remaining <= 1e-9 * demand:   # tolerance: a float residual must not open the next price level
            break
    return price, dispatch


def profits(bids):
    """Per-firm profit for one round: (p_clear - c) * q_dispatched."""
    price, dispatch = clear_market(bids)
    return (price - COST) * dispatch


# --- Payoff tables over the action grid ----------------------------------------
def encode(actions, k=N_PRICES):
    """Action profile (a_0, ..., a_{n-1}) -> single integer, base k. Used as the state index."""
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


def payoff_tables(n=N_FIRMS, k=N_PRICES, pay_as_bid=False):
    """Evaluate the market once on every action profile.

    Returns (profit, price):
      profit[encode(profile), i] = firm i's profit when `profile` is played
      price[encode(profile)]     = clearing price for `profile`
    3375 profiles x 3 firms - the whole game fits in a lookup table.
    pay_as_bid: same dispatch order, but every dispatched firm is paid its OWN bid (the
    reported `price` stays the marginal bid, so Delta remains comparable).
    """
    profit = np.zeros((k ** n, n))
    price = np.zeros(k ** n)
    for profile in itertools.product(range(k), repeat=n):
        bids = PRICES[list(profile)]
        p, dispatch = clear_market(bids)
        profit[encode(profile)] = ((bids if pay_as_bid else p) - COST) * dispatch
        price[encode(profile)] = p
    return profit, price
