import random
import sys
import typing as _t

import numpy as np
from numpy import random as npr
from sklearn.kernel_approximation import RBFSampler
from ..ch5_DP.grid_world import ACTION_SPACE, ACTIONS, GridWorld, IntVec2d, REWARDS
from ..ch5_DP.iterative_policy_evaluation_deterministic import PolicyDict, print_values
from ..ch6_MC.monte_carlo import get_policy_lazy_programmer


ALPHA: float = 0.10
GAMMA: float = 0.90
EPSILON: float = 0.10
MAX_ITERS: int = 1000


def plot_mse(mse: _t.List[float]) -> None:
    """Plot MSE against iterations."""
    from matplotlib import pyplot as plt

    plt.plot(range(len(mse)), mse)
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.show()


def choose_action_epsilon_greedy(
    policy: PolicyDict, state: IntVec2d, eps: float = EPSILON
) -> IntVec2d:
    return ACTION_SPACE[npr.randint(len(ACTION_SPACE))] if npr.random() < eps else policy[state]


def get_samples(env: GridWorld, n_episodes: int = 1000) -> _t.List[IntVec2d]:
    """Returns a list of samples of states for the environment."""
    samples = []
    for _ in range(n_episodes):
        s = (2, 0)
        samples += [s]
        while s not in env.terminal_states:
            a = random.choice(ACTION_SPACE)
            s, _ = env.act(s, a)
            samples += [s]
    return samples


class Model:
    """Models value function with RBF kernel."""

    def __init__(self, samples: _t.List[IntVec2d]) -> None:
        self._featurizer = RBFSampler()
        self._featurizer.fit(samples)
        dims = self._featurizer.random_offset_.shape[0]
        self.w = np.zeros(dims)

    def predict(self, s: IntVec2d) -> float:
        """Returns prediction of value function for state."""
        return np.dot(self.w, self._featurizer.transform([s])[0])

    def grad(self, s: IntVec2d) -> np.array:
        """Returns gradient of value function approximator for state."""
        return self._featurizer.transform([s])[0]


def main() -> int:
    env = GridWorld(3, 4, ACTIONS, REWARDS)
    samples = get_samples(env)
    model = Model(samples)
    Pi = get_policy_lazy_programmer()

    mse: _t.List[float] = []
    for _iter in range(MAX_ITERS):
        steps = 0
        err_epsd = 0.0  # Keep track of cumulative err during episode.
        s = (2, 0)  # Reset to initial state.
        Vs = model.predict(s)
        while s not in env.terminal_states:
            steps += 1
            a = choose_action_epsilon_greedy(Pi, s)
            s2, r = env.act(s, a)
            Vs2 = model.predict(s2)
            target = r if s2 in env.terminal_states else r + GAMMA * Vs2
            err = target - Vs
            err_epsd += err * err
            model.w += ALPHA * err * model.grad(s)
            s, Vs = s2, Vs2
        mse += [err_epsd / steps]  # Append Mean Sq. Err. over the episode.

    # Obtain predicted value for each state.
    V = {s: 0.0 if s in env.terminal_states else model.predict(s) for s in env.states}
    print_values(env, V)
    plot_mse(mse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
