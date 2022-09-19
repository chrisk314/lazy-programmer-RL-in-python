from collections import defaultdict
from itertools import product
import random
import sys
import typing as _t

import numpy as np
from numpy import random as npr
from ..ch5_DP.grid_world import (
    ACTION_SPACE,
    ACTIONS,
    GridWorld,
    IntVec2d,
    REWARDS,
    WindyGridWorldPenalised,
)
from ..ch5_DP.iterative_policy_evaluation_deterministic import (
    DELTA_CONV,
    PolicyDict,
    print_policy,
    print_values,
)
from ..ch6_MC.monte_carlo_control import plot_deltas, print_sample_counts


ALPHA: float = 0.10
GAMMA: float = 0.90
EPSILON: float = 0.05
MIN_ITERS: int = 1000


def choose_action_from_Q_epsilon_greedy(
    Q: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], float], s: IntVec2d, eps: float = EPSILON
) -> IntVec2d:
    """Choose action from Q given s following epsilon-greedy."""
    if npr.random() < eps:
        return ACTION_SPACE[npr.randint(len(ACTION_SPACE))]
    else:
        Q_sa = [Q[(s, a)] for a in ACTION_SPACE]
        Q_sa_max_idxs = np.argwhere(Q_sa == np.max(Q_sa))
        a_max_idx = random.choice(Q_sa_max_idxs.flat)
        return ACTION_SPACE[a_max_idx]


def get_policy_and_value_func_from_Q(
    env: GridWorld, Q: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], float]
) -> _t.Tuple[PolicyDict, _t.Dict[IntVec2d, float]]:
    V = {s: 0.0 for s in env.terminal_states}
    policy = {}
    for s in set(env.states) - set(env.terminal_states):
        a_max = ACTION_SPACE[np.argmax([Q[(s, a)] for a in ACTION_SPACE])]
        policy[s] = a_max
        V[s] = Q[(s, a_max)]
    return policy, V


def plot_rewards(rewards: _t.List[float]) -> None:
    """Plot cumulative rewards against iterations."""
    from matplotlib import pyplot as plt

    plt.plot(range(len(rewards)), rewards)
    plt.xlabel("Iterations")
    plt.ylabel("Rewards")
    plt.show()


def main() -> int:
    # print(f"Penalty term: {PENALTY}")
    # env = WindyGridWorldPenalised(PENALTY, 3, 4, ACTIONS, REWARDS)
    env = GridWorld(3, 4, ACTIONS, REWARDS)

    # Initialise Q to random values for all s and a
    Q: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], float] = {
        (s, a): 0.0 for (s, a) in product(env.states, ACTION_SPACE)
    }
    sa_cnt: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], int] = defaultdict(lambda: 0)
    delta: _t.List[float] = []
    rewards: _t.List[float] = [0.0]

    iter = 0
    while True:
        iter += 1
        max_delta = 0.0
        r_episode = 0.0

        # Reset to initial state
        s = (2, 0)
        # Choose initial action
        a = choose_action_from_Q_epsilon_greedy(Q, s)

        while s not in env.terminal_states:
            sa_cnt[(s, a)] += 1
            # Get reward and next state resulting from action
            s2, r = env.act(s, a)
            # Choose next action in new state
            a2 = choose_action_from_Q_epsilon_greedy(Q, s2)
            # Update Q value for s and a
            Q_prev = Q[(s, a)]
            Q[(s, a)] = Q_prev + ALPHA * (r + GAMMA * Q[(s2, a2)] - Q_prev)
            if (_delta := abs(Q[(s, a)] - Q_prev)) > max_delta:
                max_delta = _delta

            s, a = s2, a2
            r_episode += r

        rewards += [r_episode]
        delta += [max_delta]
        if iter > MIN_ITERS and max_delta < DELTA_CONV:
            break

    Pi, V = get_policy_and_value_func_from_Q(env, Q)
    print_policy(env, Pi)
    print_values(env, V)
    print_sample_counts(env, sa_cnt)
    plot_deltas(delta)
    plot_rewards(rewards)

    return 0


if __name__ == "__main__":
    sys.exit(main())
