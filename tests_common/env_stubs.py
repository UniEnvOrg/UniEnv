"""Extra stub environments built on top of :mod:`tests_common.envs`."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from tests_common.envs import StubFuncEnv


class ArrayStepsStubFuncEnv(StubFuncEnv):
    """:class:`StubFuncEnv` whose unbatched ``steps`` state stays a 0-d array after ``step``.

    The shared stub turns ``steps`` into a numpy scalar on unbatched ``step``,
    which its own ``reset`` then cannot assign into; this keeps it an array so
    unbatched step -> reset sequences work.
    """

    def step(self, state: Dict[str, np.ndarray], action: np.ndarray):
        out = super().step(state, action)
        new_state = dict(out[0])
        new_state["steps"] = np.asarray(new_state["steps"])
        return (new_state, *out[1:])
