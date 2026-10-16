"""ChainedTransformation caches its intermediate spaces; in-place changes must never make it stale."""
import copy
import pickle

import numpy as np
import pytest

from unienv_interface.space import BoxSpace, DictSpace, TupleSpace
from unienv_interface.space.space import Space
from unienv_interface.transformations import (
    BatchifyTransformation,
    ChainedTransformation,
    DataTransformation,
    DictExcludeKeyTransformation,
    DictIncludeKeyTransformation,
    DictTransformation,
    FlattenDictTransformation,
    IdentityTransformation,
    RescaleTransformation,
)
from unienv_interface.backends.numpy import NumpyComputeBackend
from tests_common.backends import cpu_backend_params

CPU_BACKENDS = cpu_backend_params()
B = NumpyComputeBackend


def _np(backend, x):
    return np.asarray(backend.to_numpy(x))


def _reference_transform(chain, source_space, data):
    """The original algorithm: recompute every intermediate space on each call."""
    space = source_space
    for t in chain.transformations:
        next_space = t.get_target_space_from_source(space)
        data = t.transform(space, data)
        space = next_space
    return data


def _obs_space(backend, device=None):
    f32 = backend.float32
    return DictSpace(backend, {
        "pos": BoxSpace(backend, -2.0, 2.0, f32, device=device, shape=(3,)),
        "vel": BoxSpace(backend, -10.0, 10.0, f32, device=device, shape=(3,)),
        "cam": BoxSpace(backend, 0, 255, backend.uint8, device=device, shape=(2, 2, 3)),
        "nested": DictSpace(backend, {"a": BoxSpace(backend, 0.0, 4.0, f32, device=device, shape=(1,))}, device=device),
    }, device=device)


def _make_chain():
    return ChainedTransformation([
        DictExcludeKeyTransformation(["cam"]),
        DictTransformation({
            "pos": RescaleTransformation(-1.0, 1.0),
            "nested/a": RescaleTransformation(0.0, 1.0),
        }),
        FlattenDictTransformation(),
    ])


def _assert_tree_equal(backend, a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_tree_equal(backend, a[k], b[k])
    else:
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(_np(backend, a), _np(backend, b))


class CountingIdentity(IdentityTransformation):
    """Identity transformation counting how often its target space is derived."""
    def __init__(self):
        self.calls = 0

    def get_target_space_from_source(self, source_space):
        self.__dict__["calls"] += 1  # bypass __setattr__: counting is not a config change
        return super().get_target_space_from_source(source_space)


@pytest.mark.parametrize("backend,device", CPU_BACKENDS)
def test_chained_transform_matches_uncached_algorithm(backend, device):
    space = _obs_space(backend, device)
    chain = _make_chain()
    rng = backend.random.random_number_generator(0, device=device)
    for _ in range(3):
        rng, x = space.sample(rng)
        _assert_tree_equal(backend, chain.transform(space, x), _reference_transform(chain, space, x))


def test_intermediate_spaces_are_built_once():
    counter = CountingIdentity()
    chain = ChainedTransformation([counter, FlattenDictTransformation(), IdentityTransformation()])
    space = _obs_space(B)
    _, x = space.sample(B.random.random_number_generator(0))
    for _ in range(5):
        chain.transform(space, x)
    assert counter.calls == 1
    other = _obs_space(B)  # equal but a different object: a new cache entry
    chain.transform(other, x)
    assert counter.calls == 2


def test_empty_and_single_chains():
    space = _obs_space(B)
    _, x = space.sample(B.random.random_number_generator(0))
    assert ChainedTransformation([]).transform(space, x) is x
    flat = ChainedTransformation([FlattenDictTransformation()]).transform(space, x)
    assert set(flat) == {"pos", "vel", "cam", "nested/a"}


@pytest.mark.parametrize("mutate", [
    "setitem", "spaces_setitem", "spaces_update", "spaces_reassign", "nested_setitem",
])
def test_in_place_space_mutation_invalidates_cache(mutate):
    space = _obs_space(B)
    chain = ChainedTransformation([
        DictTransformation({"pos": RescaleTransformation(-1.0, 1.0)}),
        DictTransformation({"pos": RescaleTransformation(0.0, 10.0)}),
        DictIncludeKeyTransformation(["pos"]),
    ])
    x = {"pos": np.full(3, 1.0, np.float32), "nested": {"a": np.zeros(1, np.float32)}}

    def run():
        out = chain.transform(space, x)["pos"]
        np.testing.assert_allclose(out, _reference_transform(chain, space, x)["pos"])
        return out

    np.testing.assert_allclose(run(), np.full(3, 7.5))  # [-2, 2] -> [-1, 1] -> [0, 10]
    new_pos = BoxSpace(B, 0.0, 4.0, B.float32, shape=(3,))
    if mutate == "setitem":
        space["pos"] = new_pos
    elif mutate == "spaces_setitem":
        space.spaces["pos"] = new_pos
    elif mutate == "spaces_update":
        space.spaces.update(pos=new_pos)
    elif mutate == "spaces_reassign":
        space.spaces = {**space.spaces, "pos": new_pos}
    else:
        # The changed subspace is nested: only the nested DictSpace itself is mutated
        space = DictSpace(B, {"inner": _obs_space(B)})
        chain = ChainedTransformation([
            DictTransformation({"inner/pos": RescaleTransformation(-1.0, 1.0)}),
            DictTransformation({"inner/pos": RescaleTransformation(0.0, 10.0)}),
        ])
        x = {"inner": {"pos": np.full(3, 1.0, np.float32)}}
        np.testing.assert_allclose(chain.transform(space, x)["inner"]["pos"], np.full(3, 7.5))
        space["inner"]["pos"] = new_pos
        np.testing.assert_allclose(chain.transform(space, x)["inner"]["pos"], np.full(3, 2.5))
        return
    np.testing.assert_allclose(run(), np.full(3, 2.5))  # [0, 4] -> [-1, 1] -> [0, 10]


def test_delitem_invalidates_cache():
    space = _obs_space(B)
    chain = ChainedTransformation([IdentityTransformation(), FlattenDictTransformation()])
    _, x = space.sample(B.random.random_number_generator(0))
    chain.transform(space, x)
    epoch = Space._mutation_epoch
    del space["cam"]
    assert Space._mutation_epoch > epoch
    del x["cam"]
    assert set(chain.transform(space, x)) == {"pos", "vel", "nested/a"}


def test_transformation_reconfiguration_invalidates_cache():
    space = _obs_space(B)
    rescale = RescaleTransformation(-1.0, 1.0)
    include = DictIncludeKeyTransformation(["pos"])
    chain = ChainedTransformation([DictTransformation({"pos": rescale}), DictTransformation({"pos": RescaleTransformation(0.0, 10.0)}), include])
    x = {"pos": np.full(3, 1.0, np.float32), "vel": np.ones(3, np.float32), "cam": np.zeros((2, 2, 3), np.uint8), "nested": {"a": np.zeros(1, np.float32)}}
    np.testing.assert_allclose(chain.transform(space, x)["pos"], np.full(3, 7.5))
    rescale.new_low = 0.0  # [-2, 2] -> [0, 1] -> [0, 10]
    np.testing.assert_allclose(chain.transform(space, x)["pos"], np.full(3, 7.5))
    np.testing.assert_allclose(chain.transform(space, x)["pos"], _reference_transform(chain, space, x)["pos"])
    include.enabled_keys = ["pos", "vel"]  # property setter
    assert set(chain.transform(space, x)) == {"pos", "vel"}
    chain.transformations[2] = DictIncludeKeyTransformation(["vel"])  # list mutated in place
    assert set(chain.transform(space, x)) == {"vel"}


def test_caches_do_not_bump_the_config_epoch():
    space = _obs_space(B)
    chain = _make_chain()
    _, x = space.sample(B.random.random_number_generator(0))
    chain.transform(space, x)
    epochs = (Space._mutation_epoch, DataTransformation._config_epoch)
    for _ in range(3):
        chain.transform(space, x)
    assert (Space._mutation_epoch, DataTransformation._config_epoch) == epochs


def test_box_construction_does_not_count_as_mutation():
    epoch = Space._mutation_epoch
    BoxSpace(B, -1.0, 1.0, B.float32, shape=(3,))
    DictSpace(B, {"a": BoxSpace(B, -1.0, 1.0, B.float32, shape=(3,))})
    assert Space._mutation_epoch == epoch
    t = TupleSpace(B, [BoxSpace(B, 0.0, 1.0, B.float32, shape=(1,))])
    t.spaces = (BoxSpace(B, 0.0, 2.0, B.float32, shape=(1,)),)
    assert Space._mutation_epoch == epoch + 1


def test_chained_cache_is_not_pickled():
    space = _obs_space(B)
    chain = _make_chain()
    _, x = space.sample(B.random.random_number_generator(0))
    expected = chain.transform(space, x)
    assert "_spaces_cache" in chain.__dict__
    assert "_spaces_cache" not in chain.__getstate__()
    for restored in (pickle.loads(pickle.dumps(chain)), copy.deepcopy(chain)):
        assert "_spaces_cache" not in restored.__dict__
        _assert_tree_equal(B, restored.transform(space, x), expected)


def test_dict_space_mapping_behaves_like_a_dict_and_pickles_as_one():
    space = _obs_space(B)
    assert isinstance(space.spaces, dict)
    assert space.spaces == dict(space.spaces)
    restored = pickle.loads(pickle.dumps(space))
    assert restored == space
    assert type(space.__getstate__()["spaces"]) is dict
    # Mutating the restored copy is tracked as well
    epoch = Space._mutation_epoch
    restored["pos"] = BoxSpace(B, 0.0, 1.0, B.float32, shape=(3,))
    assert Space._mutation_epoch > epoch
    assert restored != space
    # State pickled before the tracking existed (plain dict) still loads tracked
    old = DictSpace.__new__(DictSpace)
    old.__setstate__({**space.__dict__, "spaces": dict(space.spaces)})
    epoch = Space._mutation_epoch
    old.spaces["x"] = BoxSpace(B, 0.0, 1.0, B.float32, shape=(1,))
    assert Space._mutation_epoch > epoch
    copied = copy.copy(space)
    assert copied == space and copied.spaces is not space.spaces


def test_chain_under_jit_uses_cache_and_never_caches_tracers():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from unienv_interface.backends.jax import JaxComputeBackend as JB
    f32 = JB.float32
    space = DictSpace(JB, {"pos": BoxSpace(JB, -2.0, 2.0, f32, shape=(3,)), "vel": BoxSpace(JB, -1.0, 1.0, f32, shape=(2,))})
    chain = ChainedTransformation([DictTransformation({"pos": RescaleTransformation(-1.0, 1.0)}), FlattenDictTransformation()])
    x = {"pos": jnp.full(3, 1.0), "vel": jnp.zeros(2)}
    eager = chain.transform(space, x)  # warms the cache outside the trace
    jitted = jax.jit(lambda d: chain.transform(space, d))(x)
    np.testing.assert_allclose(np.asarray(jitted["pos"]), np.asarray(eager["pos"]))
    np.testing.assert_allclose(np.asarray(chain.transform(space, x)["pos"]), np.full(3, 0.5))

    # Intermediate spaces built inside a trace hold tracers (batch_space indexes the bounds)
    box = BoxSpace(JB, -1.0, 1.0, f32, shape=(3,))
    batch_chain = ChainedTransformation([BatchifyTransformation(), IdentityTransformation()])
    out = jax.jit(lambda v: batch_chain.transform(box, v))(jnp.zeros(3))
    assert out.shape == (1, 3)
    assert "_spaces_cache" not in batch_chain.__dict__
    assert batch_chain.transform(box, jnp.zeros(3)).shape == (1, 3)  # eager: builds and caches
    assert "_spaces_cache" in batch_chain.__dict__
