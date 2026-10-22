"""Protocol constants shared by client and server.

Request:   ``[id, target, method, args, kwargs]`` (``target == ""`` for protocol calls)
Response:  ``[id, OK, value]`` or ``[id, OK, value, updates]``; ``[id, ERROR, [type, message, traceback]]``
Event:     ``[subscription, EVENT, [kind, sequence, value]]``
"""

from typing import Dict, FrozenSet, Tuple

PROTOCOL_VERSION: int = 2

OK, ERROR, EVENT = 0, 1, 2

#: The only methods a client may call, per resource kind. Never generic attribute access.
METHODS: Dict[str, FrozenSet[str]] = {
    "env": frozenset({"reset", "reload", "step", "render"}),
    "world": frozenset({"step", "reset", "reload", "after_reset", "after_reload"}),
    "node": frozenset({
        "reset", "reload", "after_reset", "after_reload", "pre_environment_step", "post_environment_step",
        "set_next_action", "get_context", "get_observation", "get_reward", "get_termination",
        "get_truncation", "get_info", "render",
    }),
    "transformation": frozenset({"transform", "get_target_space_from_source"}),
}
#: Exposed only when the hosted object actually defines them.
OPTIONAL_METHODS: FrozenSet[str] = frozenset({"reload"})
#: Calls after which the server checks whether any space of the resource group changed.
RESET_METHODS: FrozenSet[str] = frozenset({"reset", "reload", "after_reset", "after_reload"})
#: Env calls whose result is published to subscribers.
PUBLISHED_METHODS: FrozenSet[str] = frozenset({"reset", "reload", "step"})

SPACES: Tuple[str, ...] = ("observation_space", "action_space", "context_space")
ATTRIBUTES: Tuple[str, ...] = (
    "name", "batch_size", "world_timestep", "world_subtimestep", "control_timestep", "update_timestep",
    "render_mode", "render_fps", "supported_render_modes", "has_reward", "has_termination_signal",
    "has_truncation_signal", "metadata",
)
PRIORITIES: Tuple[str, ...] = tuple(f"{phase}_priorities" for phase in (
    "reset", "reload", "after_reset", "after_reload", "pre_environment_step", "post_environment_step"))
