from typing import Any, Generic, Iterable, Union, Mapping, Sequence, TypeVar, Optional, Tuple, Literal
import sys
import numpy as np
from unienv_interface.backends import ComputeBackend, ArrayAPIArray
from .metadata import SpaceMetadata, EMPTY_METADATA, coerce_metadata
import abc

def _is_jax_tracer(value : Any) -> bool:
    """Return whether ``value`` is a jax tracer (a value staged inside ``jax.jit`` & co.).

    Spaces and transformations cache values derived from their (immutable)
    bounds. A tracer is only valid inside the trace that created it, so such
    caches must never store one: computing a derived value inside a trace and
    reusing it afterwards would leak the tracer.
    """
    jax = sys.modules.get("jax")  # no jax imported -> there cannot be any tracer
    return jax is not None and isinstance(value, jax.core.Tracer)

def _holds_jax_tracer(value : Any) -> bool:
    """Return whether a space (recursively, through its attributes and containers) holds a jax tracer."""
    if "jax" not in sys.modules:
        return False
    if _is_jax_tracer(value):
        return True
    if isinstance(value, Space):
        return any(_holds_jax_tracer(v) for v in value.__dict__.values())
    if isinstance(value, Mapping):
        return any(_holds_jax_tracer(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_jax_tracer(v) for v in value)
    return False

SpaceDataT = TypeVar("SpaceDataT", covariant=True)
_SpaceBDeviceT = TypeVar("_SpaceBDeviceT", covariant=True)
_SpaceBDTypeT = TypeVar("_SpaceBDTypeT", covariant=True)
_SpaceBDRNGT = TypeVar("_SpaceBDRNGT", covariant=True)
class Space(abc.ABC, Generic[SpaceDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]):
    """Abstract description of a valid data domain.

    Spaces carry backend, device, shape, and dtype metadata and define the
    operations needed by the rest of UniEnv: validation, sampling, empty value
    creation, serialization-friendly representation, and backend/device
    conversion for both the space definition and its data.

    In-place changes to a space (reassigning one of its attributes, or e.g.
    ``DictSpace.__setitem__``) increment the global ``Space._mutation_epoch``.
    Caches keyed on a space's identity (e.g. in ``ChainedTransformation``)
    include this counter, so mutating a space can never make them stale.
    ``BoxSpace`` is immutable after construction and is not tracked.

    Metadata
    --------
    Every space carries an optional, read-only ``metadata`` mapping
    (constructor keyword ``metadata=``) describing what its data *means*, e.g.
    ``BoxSpace(..., shape=(480, 640, 3), metadata={"kind": "image", "format": "rgb"})``.
    Values must be JSON-serializable; they are validated and deep-copied at
    construction and exposed as a :class:`SpaceMetadata` (empty when absent).
    Nested spaces carry their own metadata, so every camera of a ``DictSpace``
    observation can be described separately. Use :meth:`with_metadata` to get
    a copy of a space with other metadata.

    Metadata is a description, not part of the data domain: ``__eq__``,
    ``is_subspace`` and ``is_subspaceeq`` ignore it (use
    ``space_utils.metadata_utils.spaces_equal_with_metadata`` to compare it
    too). It is kept by ``to()``, the batching utilities, serialization and the
    transformations that keep the meaning of the data, and dropped where the
    meaning is lost (e.g. ``flatten_space``).

    The keys below form a shared vocabulary understood by UniEnv utilities
    (e.g. the video wrappers and ``unienv_data.storages.dict_storage_kwargs_from_space``).
    None of them is required or enforced, and any other keys are allowed:

    * ``kind``: what the data is, e.g. ``"image"``, ``"depth"``, ``"proprio"``, ``"action"``.
    * ``format``: pixel format of images, ``"rgb"``, ``"bgr"``, ``"gray"`` or ``"rgba"``.
    * ``layout``: axis order of images, ``"HWC"`` or ``"CHW"`` (trailing dims of the space).
    * ``units``: physical units, e.g. ``"m"``, ``"rad"``.
    * ``source``: name of the camera / sensor producing the data.
    * ``fps``: capture rate of the source.
    * ``label``: human readable name.
    """
    _mutation_epoch : int = 0
    # Attributes that are lazily filled caches rather than part of the space definition
    _UNTRACKED_ATTRS : frozenset = frozenset()
    # Spaces without metadata (and spaces unpickled from before metadata existed) use the class default
    _metadata : SpaceMetadata = EMPTY_METADATA

    @staticmethod
    def _notify_mutation() -> None:
        """Record that some space was changed in place (invalidates dependent caches)."""
        Space._mutation_epoch += 1

    def __setattr__(self, name : str, value : Any) -> None:
        if name in self.__dict__ and name not in self._UNTRACKED_ATTRS:
            Space._notify_mutation()
        object.__setattr__(self, name, value)

    def __init__(
        self,
        backend : ComputeBackend[ArrayAPIArray, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT],
        shape: Optional[Sequence[int]] = None,
        device : Optional[_SpaceBDeviceT] = None,
        dtype: Optional[_SpaceBDTypeT] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        # Written to ``__dict__`` directly: on a fresh space these are not mutations, and
        # this keeps the tracking ``__setattr__`` off the construction path.
        attrs = self.__dict__
        if "backend" in attrs:  # re-initialised in place
            Space._notify_mutation()
            attrs.pop("_metadata", None)
        attrs["backend"] = backend
        attrs["_shape"] = None if shape is None else tuple(shape)
        attrs["dtype"] = dtype
        attrs["_device"] = device
        if metadata is not None and metadata is not EMPTY_METADATA:
            attrs["_metadata"] = metadata if type(metadata) is SpaceMetadata else coerce_metadata(metadata)

    @property
    def device(self) -> Optional[_SpaceBDeviceT]:
        return self._device

    @property
    def metadata(self) -> SpaceMetadata:
        """Read-only description of the data (see "Metadata" in the class docstring); empty if none was given."""
        return self._metadata

    def with_metadata(
        self,
        metadata : Optional[Mapping[str, Any]] = None,
        /,
        *,
        replace : bool = False,
        **updates : Any,
    ) -> "Space[SpaceDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]":
        """Return a copy of this space with updated metadata (this space is left unchanged).

        ``metadata`` and ``updates`` are merged into the current metadata, or
        replace it entirely with ``replace=True`` (``space.with_metadata(replace=True)``
        clears it). Nested spaces are shared with this space, not copied.
        """
        merged = {} if replace else dict(self._metadata)
        if metadata is not None:
            merged.update(metadata)
        merged.update(updates)
        return self._copy_with_metadata(coerce_metadata(merged))

    def _copy_with_metadata(self, metadata : SpaceMetadata) -> "Space[SpaceDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]":
        """Shallow copy of this space with ``metadata`` (already a ``SpaceMetadata``)."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.__dict__["_metadata"] = metadata
        return new
    
    @abc.abstractmethod
    def to(
        self, 
        backend: Optional[ComputeBackend] = None,
        device: Optional[Union[_SpaceBDeviceT, Any]] = None,
    ) -> Union["Space[SpaceDataT, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]", "Space"]:
        """Return an equivalent space on another backend and/or device."""
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Return the shape of the space as an immutable property."""
        return self._shape

    @abc.abstractmethod
    def sample(self, rng : _SpaceBDRNGT, **kwargs) -> Tuple[_SpaceBDRNGT, SpaceDataT]:
        """Draw one valid value from the space and return the advanced RNG."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_empty(
        self
    ) -> SpaceDataT:
        """Create an empty data structure for this space."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_bounded(self, manner: Literal["both", "below", "above"] = "both") -> bool:
        """Return boolean specifying if this space is bounded in the specified manner."""
        raise NotImplementedError

    @abc.abstractmethod
    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        raise NotImplementedError

    def is_subspaceeq(self, other: "Space") -> bool:
        """Return whether this space is a non-strict subspace of ``other`` (``self ⊆ other``).

        A space ``a`` is a non-strict subspace of ``b`` (``a ⊆ b``) when every
        valid sample of ``a`` is also a valid member of ``b``; equality is
        allowed, i.e. ``a == b`` implies ``a.is_subspaceeq(b)``.

        The primary use case is a controller that declares its REQUIRED
        observation space and checks
        ``required.is_subspaceeq(env_observation_space)``: for ``DictSpace``
        this must hold even when the environment space exposes EXTRA keys
        beyond the required ones (unlike ``DictSpace.contains``, which demands
        exact key equality). Controller-required-space checks should typically
        use ``is_subspaceeq`` rather than the strict ``is_subspace`` because a
        controller's required space may exactly equal the env space.

        Comparison policy (structural only):

        * The two spaces must share the same backend type.
        * Dtypes must be strictly equal; no implicit cast widening is performed.
        * ``device`` is intentionally ignored — two spaces on different devices
          may still be in a subspace relation.

        Cross-type comparisons (e.g. ``BoxSpace`` vs ``DictSpace``) return
        ``False`` rather than raising. Subclasses override this method to
        provide concrete structural containment checks; the base implementation
        raises ``NotImplementedError`` to mirror the abstract-method style of
        this class.
        """
        raise NotImplementedError

    def is_subspace(self, other: "Space") -> bool:
        """Return whether this space is a STRICT subspace of ``other`` (``self ⊂ other``).

        Defined uniformly for all spaces as::

            self.is_subspace(other)  ⟺  self.is_subspaceeq(other) and not other.is_subspaceeq(self)

        I.e. ``self ⊆ other`` holds but ``other ⊆ self`` does not, so ``self``
        is a PROPER (strict) subspace of ``other``. This is the ``⊂`` relation
        versus the non-strict ``⊆`` provided by :meth:`is_subspaceeq`.

        This definition is used instead of relying on ``__eq__`` because some
        space classes only have identity ``__eq__``; defining strict
        containment via the symmetric non-strict check works uniformly for all
        classes regardless of their ``__eq__`` implementation.

        For structurally-distinct-but-mutually-containing spaces (which should
        not occur under the strict dtype/shape policies enforced by the
        per-class ``is_subspaceeq`` implementations) this degrades gracefully
        to ``False``: if both ``self.is_subspaceeq(other)`` and
        ``other.is_subspaceeq(self)`` hold, the two spaces are considered
        equivalent and neither is a STRICT subspace of the other.

        If either side's ``is_subspaceeq`` is not implemented (the base
        :meth:`is_subspaceeq` raises ``NotImplementedError``), the exception
        propagates to the caller — it is NOT swallowed into ``False`` so that
        callers can tell that the comparison is unsupported.

        Note: controller-required-space checks should typically use
        :meth:`is_subspaceeq` (a controller's required space may exactly equal
        the env space, in which case the strict ``is_subspace`` would return
        ``False``).
        """
        return self.is_subspaceeq(other) and not other.is_subspaceeq(self)

    def clip(self, x: Any) -> SpaceDataT:
        """Project ``x`` onto this space and return the projected value.

        ``clip`` is defined on the TARGET (subset) space. Contract: if
        ``self.is_subspaceeq(source)`` holds, then for every ``x`` with
        ``source.contains(x)``::

            self.contains(self.clip(x))

        unless no meaningful projection exists, in which case ``clip`` raises
        ``ValueError`` instead of inventing data (e.g. a required ``DictSpace``
        key is missing, or a string / dynamic axis / graph is shorter or
        smaller than this space's minimum; each space documents its cases).

        Additionally:

        * ``clip`` is idempotent on members: ``self.clip(y)`` equals ``y``
          whenever ``self.contains(y)``.
        * The input is never mutated; the result is new data, or ``x`` (or
          parts of it) itself where nothing needs to change.
        * The result follows this space's backend and dtype conventions (e.g.
          a ``BoxSpace`` casts to its dtype).

        Inputs that are not members of any superspace (wrong structure, wrong
        shape, ...) are not supported and may raise.

        The base implementation raises ``NotImplementedError`` to mirror the
        abstract-method style of this class; subclasses override it.
        """
        raise NotImplementedError

    def __eq__(self, other : "Space"):
        """Return boolean specifying if this space is equal to another space."""
        return self is other

    def __contains__(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return self.contains(x)
    
    def __repr__(self) -> str:
        return self.get_repr(
            abbreviate=False,
            include_backend=True,
            include_device=True,
            include_dtype=True
        )

    def __str__(self) -> str:
        return self.get_repr(
            abbreviate=True,
            include_backend=True,
            include_device=True,
            include_dtype=False
        )

    @abc.abstractmethod
    def get_repr(
        self,
        abbreviate : bool = False,
        include_backend : bool = True,
        include_device : bool = True,
        include_dtype : bool = True,
    ) -> str:
        """Return a string representation of the space."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def data_to(
        self, 
        data : SpaceDataT, 
        backend : Optional[ComputeBackend] = None,
        device : Optional[Union[_SpaceBDeviceT, Any]] = None
    ) -> Union[SpaceDataT, Any]:
        """Convert space-compatible data to another backend and/or device."""
        raise NotImplementedError

    @staticmethod
    def abbr_device(spaces : "Iterable[Space[Any, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT]]") -> Optional[_SpaceBDeviceT]:
        """Return the shared device across spaces, or ``None`` if mixed/empty."""
        
        iter_spaces = iter(spaces)
        try:
            first_space = next(iter_spaces)
        except StopIteration:
            return None
        device = first_space.device
        for space in spaces:
            if space.device != device:
                return None
        return device
