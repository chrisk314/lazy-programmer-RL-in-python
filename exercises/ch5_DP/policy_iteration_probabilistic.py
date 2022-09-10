from pprint import pprint
import sys
import typing as _t

from grid_world import ACTIONS, REWARDS, WindyGridWorld, WindyGridWorldPenalised
from iterative_policy_evaluation_deterministic import evaluate_policy, get_policy, print_policy
from iterative_policy_evaluation_probabilistic import get_transition_prob_and_rewards
from policy_iteration_deterministic import improve_policy


GAMMA: float = 0.90
PENALTY: float = -0.1


def main() -> int:
    # FIXME : `WindyGridWorldPenalised` gives strange values even with `penalty=0.0`,
    #       : for which the values should be the same as `WindyGridWorld`.
    print(f"Penalty term: {PENALTY}")
    env = WindyGridWorldPenalised(PENALTY, 3, 4, ACTIONS, REWARDS)
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
    print_policy(env, Pi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
