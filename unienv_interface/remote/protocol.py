"""Shared protocol method capabilities, independent of client and server."""

from typing import Any, Dict, List, Literal, Set, SupportsFloat, Tuple, TYPE_CHECKING, TypedDict, Union

if TYPE_CHECKING:
    from unienv_interface.backends import ArrayAPIArray

ResourceKind = Literal["env", "world", "node"]
ExecutionMode = Literal["env", "components"]
OperationKind = Literal["reset", "reload", "step"]
WireMessage = Dict[str, Any]
Descriptor = Dict[str, Any]
ResetResult = Tuple[Any, Any, Dict[str, Any]]
Reward = Union[SupportsFloat, "ArrayAPIArray"]
Signal = Union[bool, "ArrayAPIArray"]
StepResult = Tuple[Any, Reward, Signal, Signal, Dict[str, Any]]


class BoundarySnapshot(TypedDict):
    sequence: int
    revision: int
    fields: List[str]
    data: Dict[str, Any]


FIELDS: Dict[ResourceKind, Set[str]] = {
    "env": {"context", "observation", "reward", "terminated", "truncated", "info", "render"},
    "node": {"context", "observation", "reward", "terminated", "truncated", "info", "render"},
    "world": {"dt"},
}

PRIORITIES: Tuple[str, ...] = tuple(f"{phase}_priorities" for phase in
                   ("reset", "reload", "after_reset", "after_reload", "pre_environment_step", "post_environment_step"))

METHODS: Dict[ResourceKind, Dict[str, Dict[str, bool]]] = {
    "env": {
        "reset": {"mutation": True},
        "step": {"mutation": True},
        "render": {"mutation": False},
        "reload": {"mutation": True, "optional": True},
    },
    "world": {
        "step": {"mutation": True},
        "reset": {"mutation": True},
        "reload": {"mutation": True},
        "after_reset": {"mutation": True},
        "after_reload": {"mutation": True},
    },
    "node": {
        "reset": {"mutation": True},
        "reload": {"mutation": True},
        "after_reset": {"mutation": True},
        "after_reload": {"mutation": True},
        "pre_environment_step": {"mutation": True},
        "post_environment_step": {"mutation": True},
        "set_next_action": {"mutation": True},
        "get_context": {"mutation": False},
        "get_observation": {"mutation": False},
        "get_reward": {"mutation": False},
        "get_termination": {"mutation": False},
        "get_truncation": {"mutation": False},
        "get_info": {"mutation": False},
        "render": {"mutation": False},
    },
}
