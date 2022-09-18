from collections import defaultdict
from itertools import product
import random
import sys
import typing as _t

import numpy as np
from ..ch5_DP.grid_world import (
    ACTION_SPACE,
    ACTIONS,
    ActionSpace,
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


def play_episode(
    env: GridWorld,
    policy: PolicyDict,
    state: _t.Optional[IntVec2d] = None,
    action: _t.Optional[IntVec2d] = None,
) -> _t.Tuple[_t.List[IntVec2d], _t.List[IntVec2d], _t.List[float], int]:
    """Play one Monte Carlo episode."""
    s: _t.List[IntVec2d] = [state if state else get_random_state(env)]
    a: _t.List[IntVec2d] = [action if action else policy[s[0]]]
    r: _t.List[float] = [0.0]

    # For all steps after the first, choose the action from the policy.
    step: int = 0
    while step < MC_MAX_STEPS:
        s2, _r = env.act(s[step], a[step])
        s += [s2]
        r += [_r]
        step += 1
        if s2 in env.terminal_states:
            break
        a += [policy[s[step]]]

    return s, a, r, step


def improve_policy(
    Pi: PolicyDict,
    Q: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], float],
    sa_cnt: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], int],
    s: _t.List[IntVec2d],
    a: _t.List[IntVec2d],
    r: _t.List[float],
    step: int,
) -> _t.Tuple[
    PolicyDict,
    _t.Dict[_t.Tuple[IntVec2d, IntVec2d], float],
    _t.Dict[_t.Tuple[IntVec2d, IntVec2d], int],
    float,
]:
    """Improves policy."""
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
            sa_cnt[sa] += 1
            Q_prev = Q[sa]
            Q[sa] = Q_prev + (G - Q_prev) / sa_cnt[sa]

            # Update the policy with the argmax action over Q
            Q_a = [Q[(s[step], _a)] for _a in ACTION_SPACE]
            Q_a_max_idxs = np.argwhere(Q_a == np.max(Q_a))
            Pi[s[step]] = ACTION_SPACE[random.choice(Q_a_max_idxs.flat)]

            # Track biggest delta in Q for convergence check
            if (_delta := abs(Q[sa] - Q_prev)) > max_delta:
                max_delta = _delta

    return Pi, Q, sa_cnt, max_delta


def main() -> int:

    random.seed(MC_RANDOM_SEED)

    # FIXME : `WindyGridWorldPenalised` gives strange values even with `penalty=0.0`,
    #       : for which the values should be the same as `WindyGridWorld`.
    # print(f"Penalty term: {PENALTY}")
    # env = WindyGridWorldPenalised(PENALTY, 3, 4, ACTIONS, REWARDS)
    env = GridWorld(3, 4, ACTIONS, REWARDS)

    # NOTE : Often a random policy will not reach a terminal state.
    Pi = get_random_policy(env)
    print_policy(env, Pi)

    # Initialise values, states, rewards, and returns.
    Q: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], float] = {
        (s, a): 0.0 for (s, a) in product(env.states, ACTION_SPACE)
    }
    sa_cnt: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], int] = defaultdict(lambda: 0)
    delta: _t.List[float] = []

    for epsd in range(1, MC_MAX_EPISODES + 1):
        s, a, r, step = play_episode(
            env, Pi, state=get_random_state(env), action=random.choice(ACTION_SPACE)
        )
        Pi, Q, sa_cnt, max_delta = improve_policy(Pi, Q, sa_cnt, s, a, r, step)
        delta += [max_delta]
        # print(f"Episode {epsd}")
        # print_values(env, V)

    print(f"Episode {epsd}")
    print_policy(env, Pi)

    # Extract state value, V, from state-action value, Q.
    V = {_s: 0.0 for _s in set(env.states)}
    # Deterministic policy.
    V.update({_s: Q[(_s, Pi[_s])] for _s in set(env.states) - set(env.terminal_states)})
    # Stochastic policy.
    # V.update(
    #     {
    #         _s: sum([Pi[(_s, _a)] * Q[(_s, _a)] for _a in ACTION_SPACE])
    #         for _s in set(env.states) - set(env.terminal_states)
    #     }
    # )
    print_values(env, V)

    return 0


if __name__ == "__main__":
    sys.exit(main())
