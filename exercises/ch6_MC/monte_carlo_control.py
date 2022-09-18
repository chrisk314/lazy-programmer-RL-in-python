from collections import defaultdict
from itertools import product
import random
import sys
import typing as _t

import numpy as np
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
    get_policy,
    PolicyDict,
    print_policy,
    print_values,
)


GAMMA: float = 0.90
PENALTY: float = -0.0
MC_MAX_EPISODES: int = 10000
MC_MAX_STEPS: int = 100
MC_FIRST_VISIT: bool = False
MC_RANDOM_SEED: _t.Optional[int] = None


def get_random_policy(env: GridWorld) -> PolicyDict:
    policy = {s: random.choice(ACTION_SPACE) for s in set(env.states) - set(env.terminal_states)}
    return policy


def get_random_state(env: GridWorld) -> IntVec2d:
    return random.choice(list(set(env.states) - set(env.terminal_states)))


def main() -> int:

    random.seed(MC_RANDOM_SEED)

    # FIXME : `WindyGridWorldPenalised` gives strange values even with `penalty=0.0`,
    #       : for which the values should be the same as `WindyGridWorld`.
    print(f"Penalty term: {PENALTY}")
    env = WindyGridWorldPenalised(PENALTY, 3, 4, ACTIONS, REWARDS)

    # NOTE : Often a random policy will not reach a terminal state.
    Pi = get_random_policy(env)
    print_policy(env, Pi)

    # Initialise values, states, rewards, and returns.
    Q: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], float] = {
        (s, a): 0.0 for (s, a) in product(env.states, ACTION_SPACE)
    }
    G_sa_cnt: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], int] = defaultdict(lambda: 0)

    delta: _t.List[float] = []
    for epsd in range(1, MC_MAX_EPISODES + 1):
        # Play Monte Carlo episode.
        s: _t.List[IntVec2d] = [get_random_state(env)]
        r: _t.List[float] = [0.0]

        # Exploring starts method: begin with random state and action pair,
        # (s0, a0) in order to fully explore.
        a: _t.List[IntVec2d] = [random.choice(ACTION_SPACE)]
        s2, _r = env.act(s[0], a[0])
        s += [s2]
        r += [_r]

        # For all steps after the first, choose the action from the policy.
        step: int = 1
        while step < MC_MAX_STEPS:
            if s2 in env.terminal_states:
                break
            a += [Pi[s[step]]]
            s2, _r = env.act(s[step], a[step])
            s += [s2]
            r += [_r]
            step += 1

        max_delta = 0.0
        sa_seq = list(zip(s[:step], a[:step]))
        # Update state values.
        G: float = 0.0
        while step > 0:
            step -= 1
            sa = sa_seq[step]
            G = r[step + 1] + GAMMA * G
            if not (MC_FIRST_VISIT and sa in set(sa_seq[:step])):
                # Update estimate of Q
                G_sa_cnt[sa] += 1
                Q_prev = Q[sa]
                Q[sa] = Q_prev + (G - Q_prev) / G_sa_cnt[sa]

                # Update the policy with the argmax action over Q
                Q_a = [Q[(s[step], _a)] for _a in ACTION_SPACE]
                Q_a_max_idxs = np.argwhere(Q_a == np.max(Q_a))
                Pi[s[step]] = ACTION_SPACE[random.choice(Q_a_max_idxs.flat)]

                # Track biggest delta in Q for convergence check
                if (_delta := abs(Q[sa] - Q_prev)) > max_delta:
                    max_delta = _delta

        delta += [max_delta]
        # print(f"Episode {epsd}")
        # print_values(env, V)
    print(f"Episode {epsd}")
    print_policy(env, Pi)

    # TODO : How to get the values from Q?
    # print_values(env, V)

    return 0


if __name__ == "__main__":
    sys.exit(main())
