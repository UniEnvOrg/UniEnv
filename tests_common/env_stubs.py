"""Extra stub environments built on top of :mod:`tests_common.envs`.

The shared stubs turn the unbatched ``steps`` counter into a numpy scalar on
``step``, after which their own ``reset`` cannot assign into it (so an
unbatched stub cannot be reset after stepping). The ``ArraySteps*`` variants
keep it a 0-d array; use them whenever a test resets an unbatched stub after
stepping it.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from tests_common.envs import StubEnv, StubFuncEnv


class ArrayStepsStubEnv(StubEnv):
    """:class:`StubEnv` that can be reset after stepping when unbatched."""

    def step(self, action: np.ndarray):
        out = super().step(action)
        self.steps = np.asarray(self.steps)
        return out


class ArrayStepsStubFuncEnv(StubFuncEnv):
    """:class:`StubFuncEnv` that can be reset after stepping when unbatched."""

    def step(self, state: Dict[str, np.ndarray], action: np.ndarray):
        out = super().step(state, action)
        new_state = dict(out[0])
        new_state["steps"] = np.asarray(new_state["steps"])
        return (new_state, *out[1:])
