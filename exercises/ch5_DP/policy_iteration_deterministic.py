import sys
import typing as _t

from grid_world import ACTIONS, GridWorld, IntVec2d, REWARDS
from iterative_policy_evaluation_deterministic import (
    evaluate_policy,
    get_policy,
    get_transition_prob_and_rewards,
    PolicyDict,
    print_policy,
    TransProbDict,
)


GAMMA: float = 0.90


def improve_policy(
    env: GridWorld,
    policy: PolicyDict,
    trans_prob: TransProbDict,
    rewards: dict,
    values: dict,
) -> tuple[PolicyDict, bool]:
    policy_is_converged: bool = True
    for s in env.states:
        v_best: float = float("-inf")
        a_best: _t.Optional[IntVec2d] = None
        for a in env.actions.get(s, set()):
            v: float = 0.0
            for s2 in env.states:
                r: float = rewards.get((s, a, s2), 0.0)
                v += trans_prob.get((s, a, s2), 0.0) * (r + GAMMA * values[s2])
            if v > v_best:
                # Store argmax action.
                v_best = v
                a_best = a
        if policy.get(s) != a_best:
            # Update policy if better action found.
            policy_is_converged = False
            policy[s] = a_best
    return policy, policy_is_converged


def main() -> int:
    env = GridWorld(3, 4, ACTIONS, REWARDS)
    P, R = get_transition_prob_and_rewards(env)
    Pi = get_policy()
    print_policy(env, Pi)
    V: dict = {}
    iter = 0
    while True:
        iter += 1
        print(f"\nPolicy improvement {iter=}")
        V = evaluate_policy(env, Pi, P, R, initial_values=V)
        Pi, Pi_is_conv = improve_policy(env, Pi, P, R, V)
        print_policy(env, Pi)
        if Pi_is_conv:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
