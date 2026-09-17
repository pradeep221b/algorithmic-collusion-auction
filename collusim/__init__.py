"""collusim: independent Q-learners in a uniform-price electricity auction, and the tests that
separate tacit collusion from merely high prices.

    import collusim as cs
    market = cs.Market()                                   # the spec market; try demand=130 or pay_as_bid=True
    r = cs.run_seed(market, seed=0, gamma=0.95, memory=1, n_rounds=5_000_000)
    r["delta"]                                             # collusion index of the frozen policies
    cs.punishment(r["Q"], r["final_state"], market)        # M3: do rivals punish, then forgive?
    cs.temptation(r["Q"], r["final_state"], market)        # M5: profitable deviation left unplayed?
    cs.bellman_residual(r["Q"], r["final_state"], market, gamma=0.95)   # learned continuation or stale estimate?
    env = cs.AuctionEnv(market)                            # reset/step interface for your own learner
"""
__version__ = "0.1.0"

from .market import Market, encode, decode
from .agents import initial_q, train, greedy_play, run_seed, ALPHA, GAMMA, BETA
from .indicators import punishment, ablation, temptation, bellman_residual, VERDICTS
from .env import AuctionEnv

__all__ = ["Market", "encode", "decode", "initial_q", "train", "greedy_play", "run_seed",
           "ALPHA", "GAMMA", "BETA", "punishment", "ablation", "temptation", "bellman_residual",
           "VERDICTS", "AuctionEnv", "__version__"]
