"""A minimal multi-agent environment around a Market, shaped like PettingZoo's parallel API.

    env = AuctionEnv(Market(), memory=1)
    obs = env.reset(seed=0)                             # int state index: last round's action profile
    obs, rewards, terminated, truncated, info = env.step([14, 14, 3])   # one action index per firm
    rewards[i]                                          # firm i's profit this round
    info["price"], info["bids"]

Bring your own learner: anything that maps (obs, reward history) -> action index per firm.
The game is an infinite repetition, so `terminated` is always False; you choose the horizon.

ponytail: reset/step in the PettingZoo shape without the dependency, obs as an int so it drops
straight into a tabular Q. Wrap in pettingzoo.ParallelEnv with Discrete spaces if you need them.
"""
import numpy as np

from .market import Market, encode


class AuctionEnv:
    def __init__(self, market=None, memory=1):
        self.market = market or Market()
        self.memory = memory
        self.profit, self.price = self.market.tables
        self.n_agents, self.n_actions = self.market.n_firms, self.market.n_prices
        self.n_states = self.n_actions ** self.n_agents if memory else 1
        self.state = 0
        self._rng = np.random.default_rng()

    def reset(self, seed=None):
        """Start from a random first profile, as the training loop does. Returns the state index."""
        self._rng = np.random.default_rng(seed)
        first = self._rng.integers(self.n_actions, size=self.n_agents)
        self.state = encode(first, self.n_actions) if self.memory else 0
        return self.state

    def step(self, actions):
        idx = encode(actions, self.n_actions)
        rewards = self.profit[idx].copy()
        self.state = idx if self.memory else 0
        info = dict(price=float(self.price[idx]), bids=self.market.prices[list(actions)].tolist())
        return self.state, rewards, False, False, info
