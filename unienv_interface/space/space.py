from typing import Any, Generic, Iterable, Union, Mapping, Sequence, TypeVar, Optional, Tuple, Literal
import numpy as np
from unienv_interface.backends import ComputeBackend, ArrayAPIArray
import abc

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
    """
    def __init__(
        self,
        backend : ComputeBackend[ArrayAPIArray, _SpaceBDeviceT, _SpaceBDTypeT, _SpaceBDRNGT],
        shape: Optional[Sequence[int]] = None,
        device : Optional[_SpaceBDeviceT] = None,
        dtype: Optional[_SpaceBDTypeT] = None,
    ):
        self.backend = backend
        self._shape = None if shape is None else tuple(shape)
        self.dtype = dtype
        self._device = device

    @property
    def device(self) -> Optional[_SpaceBDeviceT]:
        return self._device
    
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
