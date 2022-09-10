import random
import typing as _t


IntVec2d = _t.Tuple[int, int]
ActionSpace = _t.Tuple[IntVec2d]
ActionsDict = _t.Dict[IntVec2d, _t.Set[IntVec2d]]
RewardsDict = _t.Dict[IntVec2d, float]

_U, _R, _D, _L = (-1, 0), (0, 1), (1, 0), (0, -1)
ACTION_SPACE: _t.Tuple[IntVec2d, ...] = (_U, _R, _D, _L)
ACTION_TO_STR_MAP = {(-1, 0): "U", (0, 1): "R", (1, 0): "D", (0, -1): "L"}
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
    (0, 3): 1.0,
    (1, 3): -1.0,
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
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def rewards(self) -> RewardsDict:
        return self._rewards

    @property
    def states(self) -> _t.Tuple[IntVec2d, ...]:
        return tuple(
            sorted(
                set(self._actions.keys()) | set(self._rewards.keys()),
                key=lambda x: (x[0], x[1]),
            )
        )

    @property
    def actions(self) -> ActionsDict:
        return self._actions

    def _get_next_state(self, state: IntVec2d, action: IntVec2d) -> IntVec2d:
        i, j = state
        di, dj = action
        new_state = (i + di, j + dj)
        return new_state

    def _update_state(self, state: IntVec2d, action: IntVec2d) -> IntVec2d:
        if not action in self._actions[state]:
            return state
        new_state = self._get_next_state(state, action)
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


class WindyGridWorld(GridWorld):
    """`WindyGridWorld` is a `GridWorld` with some stochastic elements.

    Extends `GridWorld`.
    """

    _trans_prob = {
        ((0, 0), _R): (((0, 1), 1.0),),
        ((0, 0), _D): (((1, 0), 1.0),),
        ((0, 1), _R): (((0, 2), 1.0),),
        ((0, 1), _L): (((0, 0), 1.0),),
        ((0, 2), _R): (((0, 3), 1.0),),
        ((0, 2), _D): (((1, 2), 1.0),),
        ((0, 2), _L): (((0, 1), 1.0),),
        ((1, 0), _U): (((0, 0), 1.0),),
        ((1, 0), _D): (((2, 0), 1.0),),
        ((1, 2), _U): (((0, 2), 0.5), ((1, 3), 0.5)),
        ((1, 2), _R): (((1, 3), 1.0),),
        ((1, 2), _D): (((2, 2), 1.0),),
        ((2, 0), _U): (((1, 0), 1.0),),
        ((2, 0), _R): (((2, 1), 1.0),),
        ((2, 1), _R): (((2, 2), 1.0),),
        ((2, 1), _L): (((2, 0), 1.0),),
        ((2, 2), _U): (((1, 2), 1.0),),
        ((2, 2), _R): (((2, 3), 1.0),),
        ((2, 2), _L): (((2, 1), 1.0),),
        ((2, 3), _U): (((1, 3), 1.0),),
        ((2, 3), _L): (((2, 2), 1.0),),
    }

    @property
    def trans_prob(self) -> _t.Dict:
        return self._trans_prob

    def _get_next_state(self, state: IntVec2d, action: IntVec2d) -> IntVec2d:
        p_s2 = self._trans_prob[(state, action)]
        new_state = random.choices(p_s2, weights=[x[1] for x in p_s2])[0][0]
        return new_state


class WindyGridWorldPenalised(WindyGridWorld):
    """`WindyGridWorldPenalised` is a `WindyGridWorld` with penalties in non-terminal states.

    Extends `WindyGridWorld`.
    """

    def __init__(self, penalty: float, *args: _t.Any, **kwargs: _t.Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self._penalise_states(penalty)

    def _penalise_states(self, penalty: float) -> None:
        if penalty > 0.0:
            raise ValueError(f"Penalty must be negative.")
        # self._default_reward = penalty
        self._rewards.update({s: penalty for s in self.states if s not in self._rewards})
