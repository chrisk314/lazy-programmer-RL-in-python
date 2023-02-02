from collections import defaultdict
import sys
import typing as _t

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
    print_values,
    StochasticPolicyDict,
)
from ..ch6_MC.monte_carlo import get_policy_lazy_programmer
from ..ch6_MC.monte_carlo_control import plot_deltas, print_sample_counts
from ..ch6_MC.monte_carlo_control_epsilon_greedy import (
    choose_action,
    get_random_stochastic_policy,
    print_stochastic_policy,
)


ALPHA: float = 0.10
GAMMA: float = 0.90
EPSILON: float = 0.05


def get_stochastic_policy_lazy_programmer() -> StochasticPolicyDict:
    det_policy: PolicyDict = get_policy_lazy_programmer()
    p_a_eps = EPSILON / len(ACTION_SPACE)
    p_a_max = 1 - EPSILON + p_a_eps

    policy = {
        s: {_a: p_a_max if _a == a else p_a_eps for _a in ACTION_SPACE}
        for s, a in det_policy.items()
    }

    return policy


def evaluate_policy(
    env: GridWorld,
    policy: StochasticPolicyDict,
    initial_values: _t.Optional[dict] = None,
) -> _t.Tuple[_t.Dict, _t.Dict[_t.Tuple[IntVec2d, IntVec2d], int], _t.List[float]]:
    """Evaluates a policy using the TD(0) scheme."""
    initial_values = initial_values or {}
    V = {s: initial_values.get(s, 0.0) for s in env.states}
    sa_cnt: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], int] = defaultdict(lambda: 0)
    delta = []
    iter = 0
    while True:
        max_delta = 0.0
        s = (2, 0)  # Reset agent to the initial state.
        step = 0
        while s not in env.terminal_states:
            a = choose_action(policy, s)
            sa_cnt[(s, a)] += 1
            s2, r = env.act(s, a)
            V_prev = V[s]
            V[s] = V_prev + ALPHA * (r + GAMMA * V[s2] - V_prev)
            if (_dv := abs(V[s] - V_prev)) > max_delta:
                max_delta = _dv
            s = s2
            step += 1
        # print(f"{iter=}")
        # print_values(env, V)
        iter += 1
        delta += [max_delta]
        if max_delta < DELTA_CONV:
            break
    print(f"Converged to tol={DELTA_CONV} in {iter} iterations.")
    return V, sa_cnt, delta


def main() -> int:
    # print(f"Penalty term: {PENALTY}")
    # env = WindyGridWorldPenalised(PENALTY, 3, 4, ACTIONS, REWARDS)
    env = GridWorld(3, 4, ACTIONS, REWARDS)

    # NOTE : Often a random policy will not reach a terminal state.
    # Pi = get_random_stochastic_policy(env)
    Pi = get_stochastic_policy_lazy_programmer()
    print_stochastic_policy(env, Pi)

    V, sa_cnt, delta = evaluate_policy(env, Pi)

    print_values(env, V)
    print_sample_counts(env, sa_cnt)
    plot_deltas(delta)

    return 0


if __name__ == "__main__":
    sys.exit(main())
