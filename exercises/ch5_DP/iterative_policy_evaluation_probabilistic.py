from pprint import pprint
import sys
import typing as _t

from grid_world import (
    _D,
    _L,
    _R,
    _U,
    ACTION_SPACE,
    ACTIONS,
    ActionsDict,
    ActionSpace,
    GridWorld,
    IntVec2d,
    REWARDS,
    WindyGridWorld,
)
import numpy as np


GAMMA: float = 0.90
DELTA_CONV: float = 1.0e-3
LARGE: float = 1.0e32
PolicyDict = _t.Dict[_t.Tuple[IntVec2d, IntVec2d], float]
TransProbDict = _t.Dict
RewardsDict = _t.Dict[_t.Tuple[IntVec2d, IntVec2d, IntVec2d], float]

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


def print_values(env: GridWorld, V: _t.Dict) -> None:
    print("V:")
    V_arr = np.zeros((env._rows, env._cols))
    for i in range(env._rows):
        for j in range(env._cols):
            V_arr[i, j] = V.get((i, j), 0.0)
    print(V_arr)


def get_transition_prob_and_rewards(env: GridWorld) -> _t.Tuple[TransProbDict, RewardsDict]:
    # TODO : `GridWorld` should implement an `Environment` interface.
    trans_prob: TransProbDict = {}
    rewards: RewardsDict = {}
    for (s, a), probs in env.trans_prob.items():
        for s2, p in probs:
            trans_prob[(s, a, s2)] = p
            rewards[(s, a, s2)] = env.rewards.get(s2, 0.0)
    return trans_prob, rewards


def get_policy() -> PolicyDict:
    return POLICY


def evaluate_policy(
    env: GridWorld, policy: PolicyDict, trans_prob: TransProbDict, rewards: _t.Dict
) -> _t.Dict:
    # TODO : `GridWorld` should implement an `Environment` interface.
    V = {s: 0.0 for s in env.states}
    iter = 0
    while True:
        max_delta = 0.0
        for s in env.states:
            v_old = V[s]
            v_new = 0.0
            for a in env.actions.get(s, set()):
                for s2 in env.states:
                    r_sas2 = rewards.get((s, a, s2), 0.0)
                    if all(
                        (
                            pi_sa := policy.get((s, a), 0.0),
                            p_sas2 := trans_prob.get((s, a, s2), 0.0),
                        )
                    ):
                        v_new += pi_sa * p_sas2 * (r_sas2 + GAMMA * V[s2])
            V[s] = v_new
            if (delta := abs(v_new - v_old)) > max_delta:
                max_delta = delta
        iter += 1
        print(f"{iter=}, {max_delta=}")
        print_values(env, V)
        if max_delta < DELTA_CONV:
            break
    return V


def main() -> int:
    env = WindyGridWorld(3, 4, ACTIONS, REWARDS)
    P, R = get_transition_prob_and_rewards(env)
    Pi = get_policy()
    V = evaluate_policy(env, Pi, P, R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
