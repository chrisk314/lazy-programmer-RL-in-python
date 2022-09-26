from collections import defaultdict
import random
import sys
import typing as _t

from gym import Env, make
from gym.core import ActType, ObsType
import numpy as np
from numpy import random as npr
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm


ALPHA: float = 0.10  # Learning rate.
GAMMA: float = 0.95  # Discount factor.
EPSILON: float = 0.05  # Exploration rate.
MAX_ITERS: int = 1000


def plot_arr_for_iters(y: _t.List[float], x_label: str = "Iterations", y_label: str = "y") -> None:
    """Plot `y` against iterations."""
    from matplotlib import pyplot as plt

    plt.plot(range(len(y)), y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def get_samples(env: Env, n_episodes: int = 1000) -> _t.List[_t.Tuple[int, ...]]:
    """Returns a list of samples of states for the environment."""
    samples = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            # TODO : Compare to `a` onehot encoded.
            samples += [[*s, a]]
            s, _, done, _, _ = env.step(a)
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
    model: Model, env: Env, s: ObsType, eps: float = EPSILON
) -> ActType:
    """Choose action from model given s following epsilon-greedy."""
    if npr.random() < eps:
        return env.action_space.sample()
    else:
        m_sa = [model.predict([*s, a]) for a in range(env.action_space.n)]
        m_sa_max_idxs = np.argwhere(m_sa == np.max(m_sa))
        a_max_idx = random.choice(m_sa_max_idxs.flat)
        return list(range(env.action_space.n))[a_max_idx]


def watch_agent(model: Model, eps: float = EPSILON) -> None:
    # NOTE : There doesn't seem to be a way to change the `Env.render_mode`
    #      : attribute after the environment has been created. Training with
    #      : rendering enabled takes too long so create a rendering env only
    #      : when required.
    env = make("CartPole-v1", render_mode="human")
    s, _ = env.reset()
    r_epsd = 0.0
    done = False
    while not done:
        a = choose_action_from_model_epsilon_greedy(model, env, s, eps=eps)
        s, r, done, _, _ = env.step(a)
        r_epsd += r
    env.close()
    print(f"Episode reward: {r_epsd}")


def main() -> int:
    env = make("CartPole-v1")
    samples = get_samples(env)
    model = Model(samples)

    mse: _t.List[float] = []
    # TODO : How best to track states visited with continuous state and action space?
    sa_cnt: _t.Dict[_t.Tuple, int] = defaultdict(lambda: 0)
    r_epsd_all: _t.List[float] = []

    watch_agent(model)  # Observe performance of agent before training

    for _iter in tqdm(range(MAX_ITERS)):
        steps = 0
        err_epsd = 0.0  # Keep track of cumulative err during episode.
        r_epsd = 0.0  # Keep track of cumulative rewards during episode
        s, _ = env.reset()  # Reset to initial state.
        done = False
        while not done:
            steps += 1
            a = choose_action_from_model_epsilon_greedy(model, env, s)
            sa_cnt[(*s, a)] += 1
            s2, r, done, _, _ = env.step(a)
            r_epsd += r
            Qsa = model.predict([*s, a])
            Qsa2 = np.max([model.predict([*s2, a2]) for a2 in range(env.action_space.n)])
            target = r if done else r + GAMMA * Qsa2
            err = target - Qsa
            err_epsd += err * err
            model.w += ALPHA * err * model.grad([*s, a])
            s = s2
        mse += [err_epsd / steps]  # Append Mean Sq. Err. over the episode.
        r_epsd_all += [r_epsd]

        if _iter > 20 and set(r_epsd_all[:-20]) == {200}:
            # Exit early if agent got max reward for last 20 iters.
            break

    plot_arr_for_iters(mse, y_label="MSE")
    plot_arr_for_iters(r_epsd_all, y_label="Rewards")
    # Observe performance of agent after training.
    # NOTE : Epsilon is set to 0.0 to observe the greedy behaviour.
    watch_agent(model, eps=0.0)

    env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
