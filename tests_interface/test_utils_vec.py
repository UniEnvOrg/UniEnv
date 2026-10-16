"""Tests for ``unienv_interface.utils.vec_util``."""

import pickle

from unienv_interface.utils.vec_util import MultiProcessFn


def test_multiprocess_fn_pickles_closures():
    offset = 10
    fn = MultiProcessFn(lambda x, *, y=0: x + y + offset)
    restored = pickle.loads(pickle.dumps(fn))
    assert restored(1, y=2) == 13
    assert fn(1) == 11
