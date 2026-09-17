"""Independent tabular Q-learning for the repeated auction.

One Q-table per agent, no shared parameters, no communication. Agents see the
public record of last round's bids (the state) and their own profit (the reward).

The training loop is compiled with numba because 10^6 rounds x 3 agents x 20+ seeds
is too slow in plain Python. The maths is exactly the textbook update:

    Q[s, a] <- (1 - alpha) Q[s, a] + alpha (r + gamma * max_a' Q[s', a'])

`run_seed(market, seed, ...)` is the one-call version: train to the horizon, freeze, measure Delta.
"""
import numpy as np
from numba import njit

from .market import N_FIRMS, N_PRICES, encode

ALPHA = 0.15
GAMMA = 0.95
BETA = 1e-5          # epsilon_t = exp(-beta t): 1.0 at t=0, 0.37 at 1e5, 0.05 at 3e5, 4.5e-5 at 1e6
N_ROUNDS = 1_000_000
CONV_ROUNDS = 100_000  # converged when no agent's greedy policy changes for this many rounds
BLOCK = 1_000          # trajectory resolution: mean price per block of rounds


def n_states(memory, n=N_FIRMS, k=N_PRICES):
    """memory=1: state is last round's full action profile (k^n states). memory=0: one state."""
    return k ** n if memory else 1


def initial_q(profit, gamma, memory, n=N_FIRMS, k=N_PRICES):
    """Calvano et al. initialisation: Q_0[s, a] = E[profit(a, rivals uniform)] / (1 - gamma).

    With rewards all >= 0, an all-zero table makes argmax pick index 0 (the lowest price)
    in every unvisited state, which biases early play toward the competitive outcome.
    Seeding every row with the discounted value of playing `a` against random rivals
    removes that bias without favouring any action over another.
    """
    S = n_states(memory, n, k)
    Q = np.zeros((n, S, k))
    table = profit.reshape((k,) * n + (n,))          # table[a0, a1, a2, i]
    for i in range(n):
        own = np.moveaxis(table[..., i], i, 0)          # own action first, rivals after
        expected = own.reshape(k, -1).mean(axis=1)      # average over all rivals' profiles
        Q[i, :, :] = expected / (1 - gamma) if gamma < 1 else expected
    return Q


@njit(cache=True)
def _argmax(row):
    best = 0
    for j in range(1, row.shape[0]):
        if row[j] > row[best]:
            best = j
    return best


@njit(cache=True)
def train(Q, profit, price, alpha, gamma, beta, n_rounds, conv_rounds, block, seed, eps0=1.0):
    """Run the repeated auction with epsilon-greedy Q-learners. Q is updated in place.

    Q       : (n_agents, n_states, n_actions), pre-initialised
    profit  : (n_actions**n_agents, n_agents) payoff table, row = encode(profile)
    price   : (n_actions**n_agents,) clearing price per profile
    eps0    : epsilon_t = eps0 * exp(-beta t); eps0 < 1 continues training from a converged Q
    Always runs the full horizon: the exploration schedule is part of the algorithm, and
    stopping early would give a single-state (memory-0) learner far less exploration than
    a 3375-state one. Convergence is recorded, not acted on.

    Returns (t_conv, streak, final_state, block_price_sum, block_count):
      t_conv       first round at which no greedy policy had changed for conv_rounds, or -1
      streak       rounds without any greedy change at the end (>= conv_rounds -> converged)
      final_state  state index after the last round played
      block_*      per-block sums and counts of the clearing price (mean = sum / count)
    """
    np.random.seed(seed)
    n, S, k = Q.shape
    memory = S > 1

    greedy = np.empty((n, S), dtype=np.int64)   # cached argmax per row so convergence is O(k)
    for i in range(n):
        for s in range(S):
            greedy[i, s] = _argmax(Q[i, s])

    a = np.empty(n, dtype=np.int64)
    for i in range(n):
        a[i] = np.random.randint(k)
    s = 0
    if memory:
        s = 0
        for i in range(n):
            s = s * k + a[i]

    n_blocks = (n_rounds + block - 1) // block
    block_sum = np.zeros(n_blocks)
    block_cnt = np.zeros(n_blocks)
    streak = 0
    t_conv = -1

    for t in range(n_rounds):
        eps = eps0 * np.exp(-beta * t)
        for i in range(n):
            if np.random.random() < eps:
                a[i] = np.random.randint(k)
            else:
                a[i] = greedy[i, s]

        profile = 0                       # index of the profile just played
        for i in range(n):
            profile = profile * k + a[i]
        s_next = profile if memory else 0   # s' IS the profile just played (memory-1)

        block_sum[t // block] += price[profile]
        block_cnt[t // block] += 1

        changed = False
        for i in range(n):
            r = profit[profile, i]
            best_next = Q[i, s_next, _argmax(Q[i, s_next])]
            Q[i, s, a[i]] = (1 - alpha) * Q[i, s, a[i]] + alpha * (r + gamma * best_next)
            g = _argmax(Q[i, s])
            if g != greedy[i, s]:
                greedy[i, s] = g
                changed = True

        streak = 0 if changed else streak + 1
        s = s_next
        if streak >= conv_rounds and t_conv < 0:
            t_conv = t

    return t_conv, streak, s, block_sum, block_cnt


def greedy_play(Q, profit, price, state, n_rounds, force=None):
    """Play the frozen greedy policies from `state` for n_rounds. No learning, no exploration.

    force: optional {round: {agent: action}} overrides, e.g. {0: {0: 0}} makes agent 0 bid
    the lowest price in the first round. Returns dict of per-round arrays.
    """
    n, S, k = Q.shape
    memory = S > 1
    force = force or {}
    actions = np.zeros((n_rounds, n), dtype=int)
    prices = np.zeros(n_rounds)
    profs = np.zeros((n_rounds, n))
    states = np.zeros(n_rounds, dtype=int)
    s = state
    for t in range(n_rounds):
        a = [int(np.argmax(Q[i, s])) for i in range(n)]
        for i, forced in force.get(t, {}).items():
            a[i] = forced
        idx = encode(a, k)
        actions[t], prices[t], profs[t], states[t] = a, price[idx], profit[idx], s
        s = idx if memory else 0
    return {"actions": actions, "prices": prices, "profits": profs, "states": states,
            "final_state": s}


def run_seed(market, seed, gamma=GAMMA, memory=1, n_rounds=N_ROUNDS, alpha=ALPHA, beta=BETA,
             eval_rounds=1_000, burn_in=100):
    """Train one seed on `market` to the horizon, then measure Delta on frozen greedy play.

    Delta is measured on what the learned policies DO, not on the noisy training trace:
    play greedy from the final state, drop `burn_in` rounds, average the clearing price.
    Returns dict(seed, Q, t_conv, converged, trajectory, final_state, mean_price, delta, last_profile).
    """
    profit, price = market.tables
    Q = initial_q(profit, gamma, memory, market.n_firms, market.n_prices)
    t_conv, streak, s, bsum, bcnt = train(Q, profit, price, alpha, gamma, beta, n_rounds,
                                          CONV_ROUNDS, BLOCK, seed)
    trajectory = np.where(bcnt > 0, bsum / np.maximum(bcnt, 1), np.nan)
    play = greedy_play(Q, profit, price, s, eval_rounds)
    mean_price = float(play["prices"][burn_in:].mean())
    return dict(seed=seed, Q=Q, t_conv=int(t_conv), converged=bool(streak >= CONV_ROUNDS),
                trajectory=trajectory, final_state=int(play["final_state"]), mean_price=mean_price,
                delta=float(market.collusion_index(mean_price)),
                last_profile=[int(x) for x in play["actions"][-1]])
