"""Trains an ML agent to play the Atari game Breakout.

Uses Deep Q learning with a convolutional neural network to process stacked
pre-processed frames.
"""

from __future__ import annotations

from collections import deque
import sys
import typing as _t

from gym import Env, make
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ..ch3_basic_RL.cart_pole_bins import plot_running_average, plot_total_rewards


ALPHA: float = 1.0e-4  # Learning rate.
GAMMA: float = 0.99  # Discount factor.
EPSILON: float = 0.05  # Exploration rate.
MAX_ITERS: int = 2000
NUM_EPISODES: int = 3000

TARGET_UPDATE_ITERS: int = 50  # Iters between updates to target model.
MINI_BATCH_SIZE: int = 32
MIN_EXPERIENCE: int = 1000
MAX_EXPERIENCE: int = 10000
FRAME_STACK_SIZE: int = 4  # Number of successive game state frames to use as agent state.

IMG_SIZE: int = 84  # Number of pixels per dimension in image frames.
# Bounding box dimensions for image transformer in `tf.image.crop_to_bounding_box` format
BREAKOUT_BOUNDING_BOX: _t.List[int] = (34, 0, 160, 160)

# TODO : Improve architecture to avoid global var.
# Global iteration counter for target network update.
GLOBAL_ITERS: int = 0


class DQN(tf.keras.Model):
    """`DQN` is a Deep Q Network for approximating the action-value function."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        conv_layer_sizes: _t.List[int],
        hidden_layer_sizes: _t.List[int],
        learning_rate: float = ALPHA,
    ) -> None:
        super().__init__()
        self._n_actions = d_out

        # Compose NN with conv and hidden layers
        self._layers: _t.List[tf.keras.Layer] = []
        for filters, kernel_size, pool_size in conv_layer_sizes:
            self._layers += [
                tf.keras.layers.Conv2D(
                    filters, kernel_size, activation=tf.nn.relu, data_format="channels_first"
                )
            ]
            self._layers += [
                tf.keras.layers.MaxPooling2D(pool_size=pool_size, data_format="channels_first")
            ]
        self._layers += [tf.keras.layers.Flatten(data_format="channels_first")]
        for d in hidden_layer_sizes:
            self._layers += [tf.keras.layers.Dense(d)]
        self._layers += [tf.keras.layers.Dense(d_out)]

        # Configure optimiser
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, x: np.ndarray) -> np.ndarray:
        _x = np.array(x, ndmin=4)  # "atleast_4d"
        for l in self._layers:
            _x = l(_x)
        return _x

    def sample_action(self, s: int, eps: float = EPSILON) -> int:
        if np.random.random() < eps:
            return np.random.choice(self._n_actions)
        return np.argmax(self(s)[0])

    def partial_fit(self, X, actions, Y):
        with tf.GradientTape() as tape:
            action_values = self(X)
            selected_action_values = tf.reduce_sum(
                action_values * tf.one_hot(actions, self._n_actions), axis=1
            )
            loss = tf.reduce_sum(tf.square(Y - selected_action_values))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))

    def train_from_buffer(self, buffer: _t.Deque, Q_t: DQN, gamma: float = GAMMA) -> None:
        if len(buffer) < MIN_EXPERIENCE:
            # Skip if not enough experience in the buffer yet.
            return
        # Sample a random batch without replacement.
        batch = buffer.sample()
        s, a, r, s2, done = zip(*batch)

        # Calculate stable returns from target network.
        G = r + np.where(~np.array(done), gamma * np.max(Q_t(s2), axis=1), 0.0)

        # Train the model on the batch.
        self.partial_fit(s, a, G)

    def copy_from(self, other) -> None:
        """Copies network parameters from another `DQN`."""
        self.set_weights(other.get_weights())


class ReplayBuffer:
    """`ReplayBuffer` stores (s0, a, r, s1, done) tuples for later sampling."""

    def __init__(
        self,
        maxlen: int = MAX_EXPERIENCE,
        batch_size: int = MINI_BATCH_SIZE,
        stack_size: int = FRAME_STACK_SIZE,
    ) -> None:
        self._maxlen: int = maxlen
        self._batch_size: int = batch_size
        self._stack_size: int = stack_size
        self._skip_idxs: _t.Set[int] = set(range(self._stack_size - 1))
        self._all_idxs: _t.Set[int] = set()
        self._data: _t.Deque = deque(maxlen=maxlen)
        self._done_idx: _t.Set = set()

    def __len__(self) -> int:
        return len(self._data)

    def append(self, item: _t.Tuple[_t.Any, ...]) -> None:
        """Adds (s0, a, r, s1, done) tuple to the buffer."""
        _len = len(self._data)
        self._data.append(item)
        if item[4]:  # if `done`
            self._done_idx |= {_len}
        if _len == self._maxlen:
            self._done_idx = {idx - 1 for idx in self._done_idx if idx > 0}
        else:
            self._all_idxs |= {_len}

    def _get_valid_idxs(self) -> _t.List[int]:
        """Returns valid buffer indices for sampling."""
        invalid_idxs = set([idx + i for i in range(1, self._stack_size) for idx in self._done_idx])
        return list(self._all_idxs - invalid_idxs - self._skip_idxs)

    def sample(self, batch_size: int = MINI_BATCH_SIZE) -> _t.List[_t.Tuple[_t.Any, ...]]:
        """Returns a sample of size `batch_size` from the buffer.

        Uniform random sampling without replacement. Special handling for edge
        cases such as episode boundaries mis sample.
        """
        if len(self._data) < batch_size:
            raise ValueError(f"Insufficient data for {batch_size=}")
        valid_idxs = self._get_valid_idxs()
        sample_idxs = np.random.choice(valid_idxs, size=batch_size, replace=False)
        # TODO : Better way to build the batch array?
        batch = [
            (
                np.array([self._data[i - j][0] for j in range(1 - self._stack_size, 1)]),
                *self._data[i][1:3],
                np.array([self._data[i - j][3] for j in range(1 - self._stack_size, 1)]),
                self._data[i][4],
            )
            for i in sample_idxs
        ]
        return batch


def show_image(data: np.ndarray, cmap: _t.Optional[str] = None, vmax: int = 255) -> None:
    """Display image to screen from numpy array."""
    plt.imshow(data, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    plt.show()


class ImageTransformer:
    """`ImageTransformer` transforms raw game state images to agent format."""

    def __init__(self, bounding_box: _t.List[int], size: _t.Tuple[int, int]) -> None:
        self._bounding_box = bounding_box
        self._size = size

    def transform(self, image: _t.Any) -> _t.Any:  # TODO : Type hints.
        """Returns game state transformed to agent format."""
        _image = tf.image.rgb_to_grayscale(image)
        _image = tf.image.crop_to_bounding_box(_image, *self._bounding_box)
        _image = tf.image.resize(_image, self._size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        _image = _image / 255  # Convert to values in [0.0, 1.0]
        _image = tf.squeeze(_image)
        return _image


class AgentState:
    def __init__(
        self, transformer: ImageTransformer, s0: _t.Any, frame_stack_size: int = FRAME_STACK_SIZE
    ) -> None:
        self._transformer: ImageTransformer = transformer
        self._max_frames: int = frame_stack_size
        self.frames = deque(
            [self._transformer.transform(s0)] * self._max_frames, maxlen=self._max_frames
        )

    @property
    def state(self) -> np.ndarray:
        return np.array(self.frames)

    def append(self, s: _t.Any):
        _s = self._transformer.transform(s)
        self.frames.append(_s)


def play_one_episode_td(
    Q: DQN,
    Q_t: DQN,
    env: Env,
    buffer: ReplayBuffer,
    img_trans: ImageTransformer,
    gamma: float = GAMMA,
    eps: float = EPSILON,
) -> float:
    global GLOBAL_ITERS

    s_raw, _ = env.reset()
    done = False
    iters = 0
    total_reward = 0.0

    # At episode start, fill the frame stack with duplicated initial frame.
    s = AgentState(img_trans, s_raw)

    # TODO : Figure out the proper way to initialise the state of `DQN` and avoid the error message below.
    # ValueError: You called `set_weights(weights)` on layer "dqn_1" with a weight list of length 6, but the layer was expecting 0 weights
    if GLOBAL_ITERS == 0:
        Q(s.state)  # Call `Q_t` to initialise state... there must be a better way...
        Q_t(s.state)  # Call `Q_t` to initialise state... there must be a better way...

    while not done and iters < MAX_ITERS:
        # Choose action using Q network and take a step
        a = Q.sample_action(s.state, eps=eps)
        s2_raw, r, done, _, _ = env.step(a)

        s.append(s2_raw)

        # Add the experience to the replay buffer
        buffer.append((s.frames[-2], a, r, s.frames[-1], done))

        # Train the agent from the experience buffer
        Q.train_from_buffer(buffer, Q_t, gamma=gamma)

        # Update iteration vars
        iters += 1
        GLOBAL_ITERS += 1
        total_reward += r

        # Update the target network after some iters
        if GLOBAL_ITERS % TARGET_UPDATE_ITERS == 0:
            Q_t.copy_from(Q)

    return total_reward


def main() -> int:
    # Set up environment and agent.
    env = make("Breakout-v0")

    # Create DQN models
    d_in = env.observation_space.shape[0]
    d_out = 4  # Action space size for Breakout is 4 not env.action_space.n=6.
    # conv sizes (num filters, kernel/filter size, pooling size)
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]
    # Q network to train
    Q = DQN(d_in, d_out, conv_layer_sizes, hidden_layer_sizes)
    # Target Q network to stabilise gradients
    Q_t = DQN(d_in, d_out, conv_layer_sizes, hidden_layer_sizes)

    # Create replay buffer with max length.
    buffer = ReplayBuffer(maxlen=MAX_EXPERIENCE)
    # Create image transformer with Breakout config
    img_trans = ImageTransformer(bounding_box=BREAKOUT_BOUNDING_BOX, size=(IMG_SIZE, IMG_SIZE))

    # Train agent.
    total_rewards = []
    for n in tqdm(range(NUM_EPISODES)):
        eps = 1.0 / np.sqrt(n + 1)  # TODO : Try exploration schedule from Minh 2013.
        total_rewards += [play_one_episode_td(Q, Q_t, env, buffer, img_trans, gamma=GAMMA, eps=eps)]
        if n == 0 or (n + 1) % 100 == 0:
            print(f"Episode {n}: total reward: {total_rewards[-1]}.")
            Q.save("breakout-dqn.tf")

    # Plot results.
    plot_total_rewards(total_rewards)
    plot_running_average(total_rewards)

    return 0


if __name__ == "__main__":
    sys.exit(main())
