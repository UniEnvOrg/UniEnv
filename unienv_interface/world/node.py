from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Set, Type, Union
from abc import ABC, abstractmethod
import numpy as np
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_interface.env_base.env import ContextType, ObsType, ActType

from .world import World

class WorldNode(ABC, Generic[ContextType, ObsType, ActType, BArrayType, BDeviceType, BDtypeType, BRNGType]):
    """A stateful node that manages one aspect of a world (e.g. a sensor, a robot, a reward function).

    The environment is initialized during the first call to ``reset`` or ``reload``.

    Lifecycle — reset flow::

        World.reset()
          -> WorldNode.reset(priority=...)
          -> WorldNode.after_reset(priority=...)
          -> WorldNode.get_context() / get_observation() / get_info() / render()

    Lifecycle — reload flow::

        World.reload()
          -> WorldNode.reload(priority=...)
          -> WorldNode.after_reset(priority=...)
          -> WorldNode.get_context() / get_observation() / get_info() / render()

    ``reload`` re-generates the simulation environment (e.g. re-reading assets,
    rebuilding the scene).  This is typically much more expensive than ``reset``
    and should only be called when the environment configuration has changed.
    By default ``reload`` delegates to ``reset``.

    Lifecycle — step flow::

        WorldNode.set_next_action(action)
          -> WorldNode.pre_environment_step(dt, priority=...)
          -> World.step()
          -> WorldNode.post_environment_step(dt, priority=...)
          -> WorldNode.get_observation()
          -> WorldNode.get_reward() / get_termination() / get_truncation() / get_info() / render()

    Implementing a node
    -------------------
    Subclasses should:

    1. Set ``name``, ``world``, and any relevant spaces / signal flags / render
       attributes (``render_mode``, ``supported_render_modes``) in ``__init__``.
    2. Override the lifecycle methods they need (``reset``, ``pre_environment_step``, etc.).
    3. **Populate the corresponding priority sets** for every lifecycle method they
       override.  Callers check these sets before dispatching a method call — a node
       will only have a given method called for priorities present in its corresponding
       set.  A node with an empty priority set for a given method will never have that
       method called.

       Example::

           class MySensor(WorldNode[...]):
               after_reset_priorities = {0}
               post_environment_step_priorities = {0}

               def post_environment_step(self, dt, *, priority=0):
                   ...  # read sensor data

       Multiple priorities (e.g. ``{0, 1}``) allow the same method to be invoked at
       different stages; the caller iterates priorities in the desired order.

    Space lifecycle contract
    ------------------------
    ``action_space``, ``observation_space``, and ``context_space`` follow a
    two-phase lifecycle:

    **Phase 1 — construction** (``__init__``):
        Nodes *should* expose a preliminary space if the dimensionality is known
        before the scene is built.  For example, an EEF controller always produces a
        6-D action regardless of the number of robot DOFs, so the space can be set to
        a ``BoxSpace`` with ``[-inf, inf]`` bounds immediately.  If the dimensionality
        is genuinely unknown (e.g. a joint-position controller whose DOF count comes
        from the compiled scene), the space may be ``None`` at this stage.

    **Phase 2 — post-build** (end of ``after_reload``):
        By the time the lowest-priority ``after_reload`` has returned, every node
        **must** have set its final, tightly-bounded spaces.  ``CombinedWorldNode``
        calls ``_refresh_spaces()`` at this point so that the aggregated spaces seen
        by the environment composer reflect the true, post-build state.
    ---
    Priority conventions (in reload / reset / after_reload / after_reset):
    Higher priority runs first (``sorted(..., reverse=True)``).

    - [100, 200) -> Floor / surrounding environment setup (e.g. table, ground plane)
    - [50, 100) -> Robot setup (add URDF/MJCF entity, init controller, apply rest pose)
    - [0, 50)   -> Object / prop setup; general post-step observation updates
    - (-, 0)    -> Sensors / renderers that must run *after* all entities are settled
                   (e.g. cameras: -50)
    """

    name : str
    control_timestep : Optional[float] = None
    update_timestep : Optional[float] = None
    context_space : Optional[Space[ContextType, BDeviceType, BDtypeType, BRNGType]] = None
    observation_space : Optional[Space[ObsType, BDeviceType, BDtypeType, BRNGType]] = None
    action_space : Optional[Space[ActType, BDeviceType, BDtypeType, BRNGType]] = None
    has_reward : bool = False
    has_termination_signal : bool = False
    has_truncation_signal : bool = False
    supported_render_modes : Sequence[str] = ()
    render_mode : Optional[str] = None
    world : Optional[World[BArrayType, BDeviceType, BDtypeType, BRNGType]] = None

    reset_priorities : Set[int] = set()
    reload_priorities : Set[int] = set()
    after_reset_priorities : Set[int] = set()
    after_reload_priorities : Set[int] = set()  # NEW: Priority set for after_reload
    pre_environment_step_priorities : Set[int] = set()
    post_environment_step_priorities : Set[int] = set()

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.world.backend

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.world.device

    @property
    def can_render(self) -> bool:
        return self.render_mode is not None

    @property
    def effective_update_timestep(self) -> Optional[float]:
        return self.update_timestep if self.update_timestep is not None else self.control_timestep

    def pre_environment_step(self, dt : Union[float, BArrayType], *, priority : int = 0) -> None:
        """
        This method is called before the environment step
        Args:
            dt (float/BArrayType): The time elapsed between the last world step and the current step (NOT the current step and next step).
        """
        pass

    def get_context(self) -> Optional[ContextType]:
        """
        Get the current context from the node.
        If the context space is None, this method should not be called.
        """
        return None

    def get_observation(self) -> ObsType:
        """
        Get the current observation from the sensor.
        If the observation space is None, this method should not be called.
        """
        raise NotImplementedError

    def get_reward(self) -> Union[float, BArrayType]:
        """
        Get the current reward from the environment.
        If `has_reward` is `False`, this method should not be called.
        """
        return 0

    def get_termination(self) -> Union[bool, BArrayType]:
        """
        Get the current termination signal from the environment.
        If `has_termination_signal` is `False`, this method should not be called.
        """
        return False
    
    def get_truncation(self) -> Union[bool, BArrayType]:
        """
        Get the current truncation signal from the environment.
        If `has_truncation_signal` is `False`, this method should not be called.
        """
        return False

    def get_info(self) -> Optional[Dict[str, Any]]:
        """
        Get optional auxiliary information with this node.
        """
        return None

    def render(self) -> Union[
        np.ndarray, BArrayType,
        Sequence[Union[np.ndarray, BArrayType]],
        Dict[str, Union[np.ndarray, BArrayType, Sequence[Union[np.ndarray, BArrayType]]]],
        None
    ]:
        """Render the current state of the node.

        If ``can_render`` is ``False``, this method should not be called.
        """
        return None

    def set_next_action(self, action: ActType) -> None:
        """
        Update the next action to be taken by the node.
        This method should be called before `pre_environment_step` call.
        If this method is not called after a world step or an action of None is given in the call, the node will compute a dummy action to try retain the same state of the robot.
        Note that if the action space is None, this method should not be called.
        """
        raise NotImplementedError

    def post_environment_step(self, dt : Union[float, BArrayType], *, priority : int = 0) -> None:
        """
        This method is called after the environment step to update the sensor's internal state.
        Args:
            dt (float/BArrayType): The time elapsed between the last world step and the current step.
        """
        pass

    def reset(
        self,
        *,
        priority : int = 0,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> None:
        """
        This method is called after `World.reset(...)` has been called.
        Reset the node and update its internal state.
        """
        pass

    def reload(
        self,
        *,
        priority : int = 0,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> None:
        """
        Reload the node. By default, this just calls `reset` with the same parameters.
        Simulation environments can override this to completely re-read assets and reload the node.
        """
        self.reset(priority=priority, seed=seed, mask=mask, **kwargs)

    def after_reset(
        self,
        *,
        priority : int = 0,
        mask : Optional[BArrayType] = None,
    ) -> None:
        """
        This method is called after all ``WorldNode``s has been called with ``reset`` (e.g. the environment reset is effectively done).
        Use ``get_context``, ``get_observation``, and ``get_info`` to read the post-reset state.
        """
        pass

    def after_reload(
        self,
        *,
        priority : int = 0,
        mask : Optional[BArrayType] = None,
    ) -> None:
        """
        This method is called after the world has been rebuilt following a reload.
        
        At this point:
        - All nodes have added their entities during the reload phase
        - The world scene has been built (simulation is ready)
        - Nodes can now cache references to entities, geoms, etc.
        
        By default, this calls ``after_reset()``. Override if you need specific
        post-reload initialization that differs from post-reset.
        
        Use ``get_context``, ``get_observation``, and ``get_info`` to read the state.
        """
        self.after_reset(priority=priority, mask=mask)

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "WorldNode":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["WorldNode"]:
        return None
    
    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the environment."""
        return hasattr(self, name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the environment."""
        return getattr(self, name)
    
    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the environment with `value`."""
        setattr(self, name, value)