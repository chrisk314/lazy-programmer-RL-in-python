import sys

from grid_world import ACTIONS, GridWorld, REWARDS, WindyGridWorldPenalised
from iterative_policy_evaluation_deterministic import (
    get_policy,
    print_policy,
    print_values,
    TransProbDict,
)
from iterative_policy_evaluation_probabilistic import get_transition_prob_and_rewards
from policy_iteration_deterministic import improve_policy


GAMMA: float = 0.90
PENALTY: float = -0.1
DELTA_CONV: float = 1.0e-3


def value_iteration(
    env: GridWorld,
    trans_prob: TransProbDict,
    rewards: dict,
    values: dict,
) -> float:
    """Performs single iteration of value iteration.

    Returns:
        Maximum delta encountered during value iteration.
    """
    max_delta: float = 0.0
    for s in set(env.states) - set(env.terminal_states):
        v_old: float = values[s]
        v_best: float = float("-inf")
        for a in env.actions.get(s, set()):
            v: float = 0.0
            for s2 in env.states:
                r: float = rewards.get((s, a, s2), 0.0)
                v += trans_prob.get((s, a, s2), 0.0) * (r + GAMMA * values[s2])
            if v > v_best:
                v_best = v
        values[s] = v_best
        if (delta := abs(v_best - v_old)) > max_delta:
            max_delta = delta
    return max_delta


def main() -> int:
    # FIXME : `WindyGridWorldPenalised` gives strange values even with `penalty=0.0`,
    #       : for which the values should be the same as `WindyGridWorld`.
    print(f"Penalty term: {PENALTY}")
    env = WindyGridWorldPenalised(PENALTY, 3, 4, ACTIONS, REWARDS)
    P, R = get_transition_prob_and_rewards(env)
    Pi = get_policy()
    print_policy(env, Pi)
    V = {s: 0.0 for s in env.states}
    iter = 0

    while True:
        iter += 1
        print(f"\nValue iteration {iter=}")
        max_delta = value_iteration(env, P, R, V)
        if abs(max_delta < DELTA_CONV):
            break
        print_values(env, V)

    # Call policy improvement once to get optimal policy
    Pi, _ = improve_policy(env, Pi, P, R, V)

    print_policy(env, Pi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
