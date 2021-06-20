#!/usr/bin/env python3
import string
from collections import deque
from pprint import pprint

import numpy as np
from numpy import random as npr

CHAR_CHOICE = list(string.ascii_lowercase + string.digits)
EPSILON = 0.05
NUM_BANDITS = 5
NUM_TRIALS = 1000


def rand_str(size: int = 4) -> str:
    """Returns a random string of fixed `size`."""
    return ''.join([npr.choice(CHAR_CHOICE) for _ in range(size)])


class Bandit(object):
    """`Bandit` represents a \"one-armed bandit\" slot machine."""

    def __init__(self):
        """Instantiates `Bandit`."""
        self.loc = npr.random()
        self.scale = npr.random()
        self._id = rand_str()

    def pull(self):
        """Returns a random outcome for the `Bandit`."""
        return npr.normal(loc=self.loc, scale=self.scale)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}-{self._id}'


if __name__ == "__main__":
    # Create a set of `Bandits`.
    bandits = [Bandit() for _ in range(NUM_BANDITS)]

    # Print some data about the bandits.
    print('Bandit, mean, stddev')
    for b in bandits:
        print(b, b.loc, b.scale)

    # Initialise `Bandit` MLE and reward state
    bandit_mle = [{'mle': 0., 'trials': 0} for b in bandits]
    rewards = deque()

    # Perform a number of trials on the set of `Bandits`.
    for t in range(NUM_TRIALS):
        # Draw random number. Choose between explore vs exploit.
        p = npr.random()
        if p < EPSILON:
            b_idx = npr.choice(NUM_BANDITS)
        else:
            b_idx = np.argmax([x['mle'] for x in bandit_mle])
        b = bandits[b_idx]
        # Make an observation
        obs = b.pull()
        rewards.append(obs)
        # Update MLE for `Bandit`.
        mle = bandit_mle[b_idx]['mle']
        mle_trials = bandit_mle[b_idx]['trials']
        bandit_mle[b_idx]['mle'] = (mle * mle_trials + obs) / (mle_trials + 1)
        bandit_mle[b_idx]['trials'] = mle_trials + 1

    avg_reward = sum(rewards) / NUM_TRIALS
    print(f'Avg reward: {avg_reward}')

    pprint(bandit_mle)

    b_idx = np.argmax([x['mle'] for x in bandit_mle])
    best_bandit = bandits[b_idx]
    print(f'Best bandit: {best_bandit}')
