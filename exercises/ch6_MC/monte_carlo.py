from collections import defaultdict
import random
import sys
from tkinter import Grid
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
    PolicyDict,
    print_policy,
    print_values,
)


GAMMA: float = 0.90
PENALTY: float = -0.0
MC_MAX_EPISODES: int = 10
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

    Pi = get_random_policy(env)
    print_policy(env, Pi)

    # Initialise values, states, rewards, and returns.
    V: dict = {s: 0 for s in env.states}
    G_s: _t.Dict[IntVec2d, _t.List[float]] = defaultdict(list)

    for epsd in range(MC_MAX_EPISODES):
        # Play Monte Carlo episode.
        s: _t.List[IntVec2d] = [get_random_state(env)]
        r: _t.List[float] = [0.0]

        step: int = 0
        while step < MC_MAX_STEPS:
            a = Pi[s[step]]
            s2, _r = env.act(s[step], a)
            s += [s2]
            r += [_r]
            step += 1
            if s2 in env.terminal_states:
                break

        # Update state values.
        G: float = 0.0
        while step > 0:
            step -= 1
            G = r[step + 1] + GAMMA * G
            G_s[s[step]] += [G]
            V[s[step]] = np.mean(G_s[s[step]])
            # if not (MC_FIRST_VISIT and s[step] in set(s[:step])):
            #     G_s[s[step]] += [G]
            #     V[s[step]] = np.mean(G_s[s[step]])

        print(f"Episode {epsd}")
        print_values(env, V)

    return 0


if __name__ == "__main__":
    sys.exit(main())
