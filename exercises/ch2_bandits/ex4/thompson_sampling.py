#!/usr/bin/env python3
"""This script explores the Thompson Sampling or Baysian Bandit
algorithm for controlling the explore vs exploit decision process.
"""
from pprint import pprint
import string
from typing import List, Tuple

from matplotlib import pyplot as plt
import nptyping as npt
import numpy as np
from numpy import random as npr
from scipy import stats


CHAR_CHOICE = list(string.ascii_lowercase + string.digits)
NUM_TRIALS = 10000
SEED = 0


def rand_str(size: int = 4) -> str:
    """Returns a random string of fixed `size`."""
    return "".join([npr.choice(CHAR_CHOICE) for _ in range(size)])


class Bandit(object):
    """`Bandit` represents a \"one-armed bandit\" slot machine."""

    def __init__(self, p: float = 1.0) -> None:
        """Instantiates `Bandit`."""
        self.p = p
        self._id = rand_str()

    def pull(self) -> float:
        """Returns a random outcome for the `Bandit`."""
        return float(int(npr.random() < self.p))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}-{self._id}"


class MLE(object):
    """`MLE` stores and updates Maximum Likelihood Estimate."""

    def __init__(self, a: float = 1.0, b: float = 1.0, trials_init: int = 0) -> None:
        self.a = a
        self.b = b
        self.trials = trials_init

    def sample(self) -> float:
        """Returns sample from Beta distribution with estimated parameters."""
        return npr.beta(self.a, self.b)

    def update(self, obs: float) -> Tuple[float, float]:
        """Updates MLE with an observation and returns new value."""
        self.a += obs
        self.b += 1 - obs
        self.trials += 1
        return self.a, self.b

    def __str__(self) -> str:
        return f"{repr(self)}: a: {self.a:1.3e}, b: {self.b:1.3e}, trials: {self.trials}"


def plot_samples(bandits: List[Bandit], mles: List[MLE], trial: int) -> None:
    """Plot sample distributions after specified number of trials."""
    x = np.linspace(0, 1, 100)
    for b, mle in zip(bandits, mles):
        y = stats.beta.pdf(x, mle.a, mle.b)
        win_rate = (mle.a - 1.0) / mle.trials if mle.trials > 0 else 0.0
        plt.plot(x, y, label=f"p: {b.p}, win rate: {win_rate}")
    plt.legend()
    plt.title(f"Distributions after {trial} trials")
    plt.show()


def run_experiment(
    bandits: List[Bandit], num_trials: int
) -> Tuple[npt.NDArray[(NUM_TRIALS,), npt.Float64], List[MLE]]:
    """Runs an experiment on a set of bandits.

    Args:
        bandits: Bandits to extract rewards from.
        alpha: Hyperparameter for UCB1 controlling exploration.
        num_trials: Number of observations to make of the bandits.

    Returns:
        Tuple containing the reward vector for all trials and a
            list of maximum likelihood estimates for the bandits.
    """
    # Initialise `Bandit` MLE and reward state
    bandit_mle: List[MLE] = [MLE() for _ in bandits]
    rewards: npt.NDArray = np.empty(num_trials)

    n_bandits = len(bandits)
    b_opt_idx = np.argmax([b.p for b in bandits])
    n_opt, n_subopt = 0, 0

    plot_iters = set([1, 5, 10, 20, 50, 100, 500, 1000, 5000])

    # Perform a number of trials on the set of `Bandits`.
    for i in range(n_bandits, num_trials):
        # Pick a bandit based on the UCB1 algorithm
        b_idx = np.argmax([x.sample() for x in bandit_mle])
        b = bandits[b_idx]
        n_opt, n_subopt = (n_opt + 1, n_subopt) if b_idx == b_opt_idx else (n_opt, n_subopt + 1)
        # Make an observation
        obs = b.pull()
        rewards[i] = obs
        bandit_mle[b_idx].update(obs)

        if i in plot_iters:
            plot_samples(bandits, bandit_mle, i)

    print(
        f"num_trials: {num_trials}, num_opt: {n_opt}, num_subopt: {n_subopt}, opt_ratio: {n_opt/num_trials}"
    )
    return rewards, bandit_mle


def plot_rewards(rewards: List[float]) -> None:
    """Plot rewards for current epsilon."""
    cum_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.plot(cum_rewards)


def display_plot() -> None:
    """Helper method to configure plot and display."""
    plt.xlabel("iters")
    plt.ylabel("average reward")
    plt.xscale("log")
    plt.grid(which="both")
    plt.show()


if __name__ == "__main__":
    # Create a set of `Bandits`.
    bandit_config = [0.1, 0.5, 0.8]
    bandits = [Bandit(p=p) for p in bandit_config]

    # Print some data about the bandits.
    print("Bandit, p")
    for b in bandits:
        print(b, b.p)

    rewards, bandit_mle = run_experiment(bandits, NUM_TRIALS)
    avg_reward = sum(rewards) / NUM_TRIALS
    print(f"Avg reward: {avg_reward}")

    pprint([str(mle) for mle in bandit_mle])

    b_idx = np.argmax([x.sample() for x in bandit_mle])
    best_bandit = bandits[b_idx]
    print(f"Best bandit: {best_bandit}")

    plot_rewards(rewards)
    display_plot()
