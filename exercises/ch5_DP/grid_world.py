import sys
import typing as _t


IntVec2d = _t.Tuple[int, int]
ActionSpace = _t.Tuple[IntVec2d]
ActionsDict = _t.Dict[IntVec2d, _t.Set[IntVec2d]]
RewardsDict = _t.Dict[IntVec2d, float]

_U, _R, _D, _L = (-1, 0), (0, 1), (1, 0), (0, -1)
ACTION_SPACE: _t.Tuple[IntVec2d, ...] = (_U, _R, _D, _L)
# Grid world coords (-y, x) with (0, 0) at top left corner
ACTIONS: ActionsDict = {
    (0, 0): set([_R, _D]),
    (0, 1): set([_R, _L]),
    (0, 2): set([_R, _D, _L]),
    (1, 0): set([_U, _D]),
    (1, 2): set([_U, _R, _D]),
    (2, 0): set([_U, _R]),
    (2, 1): set([_R, _L]),
    (2, 2): set([_U, _R, _L]),
    (2, 3): set([_U, _L]),
}
REWARDS: RewardsDict = {
    (0, 3): 1,
    (1, 3): -1,
}


class GridWorld:
    """`GridWorld` represents a 2D grid environment for an RL agent."""

    _default_reward: float = 0.0

    def __init__(self, rows: int, cols: int, actions: ActionsDict, rewards: RewardsDict) -> None:
        """Instantiates `GridWorld`.

        Args:
            rows: Number of rows in the grid (y axis).
            cols: Number of columns in the grid (x axis).
            actions: Possible actions for the agent.
            rewards: Rewards for the states.
        """
        self._rows: int = rows
        self._cols: int = cols
        self._actions: ActionsDict = actions
        self._rewards: RewardsDict = rewards

    @property
    def states(self) -> _t.Tuple[IntVec2d, ...]:
        return tuple(
            sorted(
                tuple(self._actions.keys()) + tuple(self._rewards.keys()),
                key=lambda x: (x[0], x[1]),
            )
        )

    @property
    def actions(self) -> ActionsDict:
        return self._actions

    def _update_state(self, state: IntVec2d, action: IntVec2d) -> IntVec2d:
        if not action in self._actions[state]:
            return state
        i, j = state
        di, dj = action
        new_state = (i + di, j + dj)
        in_x = 0 <= new_state[0] < self._rows
        in_y = 0 <= new_state[1] < self._cols
        if in_x and in_y:
            return new_state
        raise ValueError("Action takes _state out of bounds.")

    def act(self, state: IntVec2d, action: IntVec2d) -> _t.Tuple[IntVec2d, float]:
        """Takes specified `action` and returns new _state and reward."""
        new_state = self._update_state(state, action)
        reward = self._rewards.get(new_state, self._default_reward)
        return new_state, reward

    def is_active(self, state: IntVec2d) -> bool:
        """Returns True iff the episode is active."""
        return state in self._actions.keys()


def policy(state: IntVec2d) -> IntVec2d:
    """Returns action to take based on state."""
    pass


def main() -> int:
    """Entrypoint for RL run."""
    E = GridWorld(3, 4, ACTIONS, REWARDS)
    s = (2, 0)
    while E.is_active(s):
        a = policy(s)
        s, r = E.act(s, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
