from collections import defaultdict
import random
import sys
import typing as _t

import numpy as np
from numpy import random as npr
from sklearn.kernel_approximation import RBFSampler
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
    print_policy,
    print_values,
)
from ..ch6_MC.monte_carlo_control import plot_deltas, print_sample_counts


ALPHA: float = 0.10  # Learning rate.
GAMMA: float = 0.90  # Discount factor.
EPSILON: float = 0.10  # Exploration rate.
MAX_ITERS: int = 1000
PENALTY: float = -0.1  # Penalty per step for penalised grid world.


def plot_mse(mse: _t.List[float]) -> None:
    """Plot MSE against iterations."""
    from matplotlib import pyplot as plt

    plt.plot(range(len(mse)), mse)
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.show()


def get_samples(env: GridWorld, n_episodes: int = 1000) -> _t.List[_t.Tuple[int, ...]]:
    """Returns a list of samples of states for the environment."""
    samples = []
    for _ in range(n_episodes):
        s = (2, 0)
        while s not in env.terminal_states:
            a = random.choice(ACTION_SPACE)
            # TODO : Compare to `a` onehot encoded.
            samples += [s + a]
            # TODO : What to store in terminal state as there is no action?
            s, _ = env.act(s, a)
    return samples  # type: ignore


class Model:
    """Models value function with RBF kernel."""

    def __init__(self, samples: _t.List[_t.Tuple]) -> None:
        self._featurizer = RBFSampler()
        self._featurizer.fit(samples)
        dims = self._featurizer.random_offset_.shape[0]
        self.w = np.zeros(dims)

    def predict(self, s: _t.Tuple) -> float:
        """Returns prediction of value function for state."""
        return np.dot(self.w, self._featurizer.transform([s])[0])

    def grad(self, s: _t.Tuple) -> np.array:
        """Returns gradient of value function approximator for state."""
        return self._featurizer.transform([s])[0]


def choose_action_from_model_epsilon_greedy(
    model: Model, s: IntVec2d, eps: float = EPSILON
) -> IntVec2d:
    """Choose action from model given s following epsilon-greedy."""
    if npr.random() < eps:
        return ACTION_SPACE[npr.randint(len(ACTION_SPACE))]
    else:
        m_sa = [model.predict(s + a) for a in ACTION_SPACE]
        m_sa_max_idxs = np.argwhere(m_sa == np.max(m_sa))
        a_max_idx = random.choice(m_sa_max_idxs.flat)
        return ACTION_SPACE[a_max_idx]


def main() -> int:
    # env = GridWorld(3, 4, ACTIONS, REWARDS)
    env = WindyGridWorldPenalised(PENALTY, 3, 4, ACTIONS, REWARDS)
    samples = get_samples(env)
    model = Model(samples)

    mse: _t.List[float] = []
    sa_cnt: _t.Dict[_t.Tuple[IntVec2d, IntVec2d], int] = defaultdict(lambda: 0)

    for _iter in range(MAX_ITERS):
        steps = 0
        err_epsd = 0.0  # Keep track of cumulative err during episode.
        s = (2, 0)  # Reset to initial state.
        while s not in env.terminal_states:
            steps += 1
            a = choose_action_from_model_epsilon_greedy(model, s)
            sa_cnt[(s, a)] += 1
            s2, r = env.act(s, a)
            Qsa = model.predict(s + a)
            Qsa2 = np.max([model.predict(s2 + a2) for a2 in ACTION_SPACE])
            target = r if s2 in env.terminal_states else r + GAMMA * Qsa2
            err = target - Qsa
            err_epsd += err * err
            model.w += ALPHA * err * model.grad(s + a)
            s = s2
        mse += [err_epsd / steps]  # Append Mean Sq. Err. over the episode.

    plot_mse(mse)

    # Obtain predicted value and greedy policy for each state.
    V, Pi = {}, {}
    for s in env.states:
        if s in env.terminal_states:
            V[s] = 0.0
        else:
            values = [model.predict(s + a) for a in ACTION_SPACE]
            V[s] = np.max(values)
            Pi[s] = ACTION_SPACE[np.argmax(values)]
    print_values(env, V)
    # TODO : How to improve exploration for infrequently visited states?
    print_sample_counts(env, sa_cnt)
    print_policy(env, Pi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
