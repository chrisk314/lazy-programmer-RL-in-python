from pprint import pprint
import sys
import typing as _t

from grid_world import _D, _L, _R, _U, ACTIONS, GridWorld, REWARDS, WindyGridWorld
from iterative_policy_evaluation_deterministic import (
    evaluate_policy,
    get_policy,
    PolicyDict,
    RewardsDict,
    TransProbDict,
)
import numpy as np


GAMMA: float = 0.90

# Policy is now probabilistic for cell (2, 0)
POLICY: PolicyDict = {
    ((0, 0), _R): 1.0,
    ((0, 1), _R): 1.0,
    ((0, 2), _R): 1.0,
    ((1, 0), _U): 1.0,
    ((1, 2), _U): 1.0,
    ((2, 0), _U): 0.5,
    ((2, 0), _R): 0.5,
    ((2, 1), _R): 1.0,
    ((2, 2), _U): 1.0,
    ((2, 3), _L): 1.0,
}


def get_transition_prob_and_rewards(env: GridWorld) -> _t.Tuple[TransProbDict, RewardsDict]:
    # TODO : `GridWorld` should implement an `Environment` interface.
    trans_prob: TransProbDict = {}
    rewards: RewardsDict = {}
    for (s, a), probs in env.trans_prob.items():
        for s2, p in probs:
            trans_prob[(s, a, s2)] = p
            rewards[(s, a, s2)] = env.rewards.get(s2, 0.0)
    return trans_prob, rewards


def main() -> int:
    env = WindyGridWorld(3, 4, ACTIONS, REWARDS)
    P, R = get_transition_prob_and_rewards(env)
    Pi = get_policy()
    V = evaluate_policy(env, Pi, P, R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
