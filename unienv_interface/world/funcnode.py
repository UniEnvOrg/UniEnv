from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Set, Type, Union
from abc import ABC, abstractmethod
import numpy as np
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space
from unienv_interface.env_base.env import ContextType, ObsType, ActType

from .funcworld import FuncWorld, WorldStateT

NodeStateT = TypeVar("NodeStateT")

class FuncWorldNode(ABC, Generic[
    WorldStateT, NodeStateT,
    ContextType, ObsType, ActType, BArrayType, BDeviceType, BDtypeType, BRNGType
]):
    """A functional (stateless) node that manages one aspect of a world.

    Unlike ``WorldNode``, every method receives and returns explicit state objects
    (``world_state``, ``node_state``) instead of mutating internal attributes.

    ``initial`` creates the node state for the first time (the environment has not
    been set up yet).  ``reset`` is called on subsequent episode boundaries when the
    node state already exists.

    Lifecycle — initial flow (environment not yet created)::

        FuncWorldNode.initial(world_state, priority=...)
          -> returns (world_state, node_state)

    Lifecycle — reset flow::

        FuncWorld.reset()
          -> FuncWorldNode.reset(world_state, node_state, priority=...)
          -> FuncWorldNode.after_reset(world_state, node_state, priority=...)
          -> FuncWorldNode.get_context(...) / get_observation(...) / get_info(...) / render(...)

    Lifecycle — reload flow::

        FuncWorld.reload()          # World prepares (clear, load assets)
          -> FuncWorldNode.reload(world_state, priority=...)     # Nodes add entities
        FuncWorld.after_reload()    # World compiles scene
          -> FuncWorldNode.after_reload(world_state, node_state, priority=...)  # Nodes cache refs
          -> FuncWorldNode.get_context(...) / get_observation(...) / get_info(...) / render(...)

    ``reload`` re-generates the simulation environment (e.g. re-reading assets,
    rebuilding the scene).  This is typically much more expensive than ``reset``
    and should only be called when the environment configuration has changed.
    By default ``reload`` delegates to ``initial``.

    Lifecycle — step flow::

        FuncWorldNode.set_next_action(world_state, node_state, action)
          -> FuncWorldNode.pre_environment_step(world_state, node_state, dt, priority=...)
          -> FuncWorld.step()
          -> FuncWorldNode.post_environment_step(world_state, node_state, dt, priority=...)
          -> FuncWorldNode.get_observation(world_state, node_state)
          -> FuncWorldNode.get_reward(...) / get_termination(...) / get_truncation(...) / get_info(...) / render(...)

    Implementing a node
    -------------------
    Subclasses should:

    1. Set ``name``, ``world``, and any relevant spaces / signal flags / render
       attributes (``render_mode``, ``supported_render_modes``) in ``__init__``
       (or as class-level attributes).
    2. Implement ``initial`` and ``reset`` (both abstract), plus any other lifecycle
       methods the node needs (``pre_environment_step``, ``post_environment_step``, etc.).
    3. **Populate the corresponding priority sets** for every lifecycle method they
       implement.  Callers check these sets before dispatching a method call — a node
       will only have a given method called for priorities present in its corresponding
       set.  A node with an empty priority set for a given method will never have that
       method called.

       Example::

           class MySensor(FuncWorldNode[...]):
               initial_priorities = {0}
               reset_priorities = {0}
               after_reset_priorities = {0}
               post_environment_step_priorities = {0}

               def initial(self, world_state, *, priority=0, seed=None, **kwargs):
                   ...
               def reset(self, world_state, node_state, *, priority=0, seed=None, mask=None, **kwargs):
                   ...

       Multiple priorities (e.g. ``{0, 1}``) allow the same method to be invoked at
       different stages; the caller iterates priorities in the desired order.

    Space contract
    --------------
    ``action_space``, ``observation_space``, and ``context_space`` are **static
    interface metadata**.  They must be fully declared in ``__init__`` and must
    not change after construction.  ``CombinedFuncWorldNode`` reads them once at
    construction time; there is no refresh mechanism.

    Unlike stateful ``WorldNode`` subclasses (which may need to build a simulation
    scene before knowing DOF counts), functional nodes are expected to derive their
    spaces purely from their configuration parameters.
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
    world : Optional[FuncWorld[WorldStateT, BArrayType, BDeviceType, BDtypeType, BRNGType]] = None

    initial_priorities : Set[int] = set()
    reset_priorities : Set[int] = set()
    reload_priorities : Set[int] = set()
    after_reset_priorities : Set[int] = set()
    after_reload_priorities : Set[int] = set()
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

    @abstractmethod
    def initial(
        self,
        world_state : WorldStateT,
        *,
        priority : int = 0,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[WorldStateT, NodeStateT]:
        raise NotImplementedError

    def reload(
        self,
        world_state : WorldStateT,
        *,
        priority : int = 0,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        Reload the node. By default, this just calls `initial` with the same parameters.
        Simulation environments can override this to completely re-read assets and reload the node.
        """
        return self.initial(world_state, priority=priority, seed=seed, **kwargs)

    @abstractmethod
    def reset(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        *,
        priority : int = 0,
        seed : Optional[int] = None,
        mask : Optional[BArrayType] = None,
        **kwargs
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        This method is called after `FuncWorld.reset(...)` has been called.
        """
        return world_state, node_state

    def after_reset(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        *,
        priority : int = 0,
        mask : Optional[BArrayType] = None
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        This method is called after all ``FuncWorldNode``s has been called with ``reset`` (e.g. the environment reset is effectively done).
        Use ``get_context``, ``get_observation``, and ``get_info`` to read the post-reset state.
        """
        return world_state, node_state

    def after_reload(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        *,
        priority : int = 0,
        mask : Optional[BArrayType] = None
    ) -> Tuple[WorldStateT, NodeStateT]:
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
        return self.after_reset(world_state, node_state, priority=priority, mask=mask)

    def pre_environment_step(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        dt : Union[float, BArrayType],
        *,
        priority : int = 0
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        This method is called before each environment step.
        Args:
            world_state (WorldStateT): The current state of the world.
            node_state (NodeStateT): The current state of the node.
            dt (Union[float, BArrayType]): The time delta since the last step.
        """
        return world_state, node_state

    def get_context(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> Optional[ContextType]:
        """
        Get the current context from the node.
        If the context space is None, this method should not be called.
        """
        return None

    def get_observation(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> ObsType:
        """
        Get the current observation from the sensor.
        If the observation space is None, this method should not be called.
        """
        raise NotImplementedError

    def get_reward(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> Union[float, BArrayType]:
        """
        Get the current reward from the environment.
        If the reward space is None, this method should not be called.
        """
        raise NotImplementedError
    
    def get_termination(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> Union[bool, BArrayType]:
        """
        Get the current termination status from the environment.
        If the termination space is None, this method should not be called.
        """
        raise NotImplementedError
    
    def get_truncation(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> Union[bool, BArrayType]:
        """
        Get the current truncation status from the environment.
        If the truncation space is None, this method should not be called.
        """
        raise NotImplementedError
    
    def get_info(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current info from the environment.
        """
        return None

    def render(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
    ) -> Union[
        np.ndarray, BArrayType,
        Sequence[Union[np.ndarray, BArrayType]],
        Dict[str, Union[np.ndarray, BArrayType, Sequence[Union[np.ndarray, BArrayType]]]],
        None
    ]:
        """Render the current state of the node.

        If ``can_render`` is ``False``, this method should not be called.
        """
        return None
    
    def set_next_action(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        action : ActType,
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        Update the next action to be taken by the node.
        This method should be called before `pre_environment_step` call.
        If this method is not called after a world step or an action of None is given in the call, the node will compute a dummy action to try retain the same state of the robot.
        Note that if the action space is None, this method should not be called.
        """
        raise NotImplementedError
    
    def post_environment_step(
        self,
        world_state : WorldStateT,
        node_state : NodeStateT,
        dt : Union[float, BArrayType],
        *,
        priority : int = 0
    ) -> Tuple[WorldStateT, NodeStateT]:
        """
        This method is called after the environment step to update the sensor's internal state.
        Args:
            world_state (WorldStateT): The current state of the world.
            node_state (NodeStateT): The current state of the node.
            dt (Union[float, BArrayType]): The time delta since the last step.
        """
        return world_state, node_state

    def close(self, world_state : WorldStateT, node_state : NodeStateT) -> WorldStateT:
        return world_state
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncWorldNode":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["FuncWorldNode"]:
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