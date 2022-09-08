from pprint import pprint
import sys
import typing as _t

from grid_world import ACTIONS, REWARDS, WindyGridWorldPenalised
from iterative_policy_evaluation_deterministic import (
    evaluate_policy,
    GAMMA,
    get_policy,
    get_transition_prob_and_rewards,
    PolicyDict,
    print_values,
    TransProbDict,
)
from policy_iteration_deterministic import improve_policy


def main() -> int:
    penalty = -0.0
    env = WindyGridWorldPenalised(penalty, 3, 4, ACTIONS, REWARDS)
    P, R = get_transition_prob_and_rewards(env)
    Pi = get_policy()
    V: dict = {}
    iter = 0
    while True:
        iter += 1
        print(f"\nPolicy improvement {iter=}")
        V = evaluate_policy(env, Pi, P, R, initial_values=V)
        Pi, Pi_is_conv = improve_policy(env, Pi, P, R, V)
        if Pi_is_conv:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
