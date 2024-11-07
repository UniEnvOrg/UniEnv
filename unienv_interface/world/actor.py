from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Protocol, Iterable, Union
from abc import ABC, abstractmethod
from ..space import Space
from ..space.dict import Dict as DictSpace
from ..backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from ..env_base.funcenv import FuncEnvCommonState, FuncEnv, StateType
from .sensor import Sensor, FuncSensor
from .world import FuncWorld
from dataclasses import dataclass

ActorMixinT = TypeVar("ActorMixinT")
ActorMixinFuncT = TypeVar("ActorMixinFuncT")
class ActorMixin(Protocol, Generic[ActorMixinT, ActorMixinFuncT]):
    """
    Actor Mixins define standardized interfaces for adding additional functionality to actors.
    For each mixin, the actor should have a dictionary of action / observation spaces
    And the mixin should have its action / observation contained inside the actor's action / observation space, with the `mixin_name` as the key.
    """

    mixin_name : str
    mixin_actor_type : Optional[Type[ActorMixinT]]
    mixin_func_actor_type : Optional[Type[ActorMixinFuncT]]

    def get_mixin_observation_space(
        self,
        instance : Union[ActorMixinT, ActorMixinFuncT],
        backend : ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType], 
        device : BDeviceType
    ) -> Optional[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]]:
        raise NotImplementedError

    def get_mixin_action_space(
        self,
        instance : Union[ActorMixinT, ActorMixinFuncT],
        backend : ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType],
        device : BDeviceType
    ) -> Optional[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]]:
        raise NotImplementedError
    
    def read_mixin_data(
        self,
        instance : ActorMixinT,
    ) -> Dict[str, Any]:
        raise NotImplementedError
    
    def read_mixin_data_func(
        self,
        instance : ActorMixinFuncT,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : "ActorStateT",
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        "ActorStateT",
        Dict[str, Any]
    ]:
        raise NotImplementedError

    def apply_mixin_action(
        self,
        instance : ActorMixinT,
        action : Any
    ) -> None:
        raise NotImplementedError
    
    def apply_mixin_action_func(
        self,
        instance : ActorMixinFuncT,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : "ActorStateT",
        action : Any,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        "ActorStateT"
    ]:
        raise NotImplementedError

    @classmethod
    def is_actor_mixin_property(cls, name : str) -> bool:
        cls._build_actor_mixin_properties()
        return name in cls._actor_mixin_properties
    
    @classmethod
    def is_actor_attachment_supported(cls, actor : "Actor") -> bool:
        cls._build_actor_mixin_properties()
        for prop in cls._actor_mixin_properties:
            if not actor.has_wrapper_attr(prop):
                return False
        return True

    @classmethod
    def _build_actor_mixin_properties(cls) -> None:
        if not hasattr(cls, "_actor_mixin_properties"):
            properties_actor = vars(cls.mixin_actor_type).keys() if cls.mixin_actor_type is not None else []
            properties_actor = [p for p in properties_actor if not p.startswith("_")]
            cls._actor_mixin_properties = properties_actor

    @classmethod
    def is_func_actor_mixin_property(cls, name : str) -> bool:
        cls._build_func_actor_mixin_properties()
        return name in cls._func_actor_mixin_properties
    
    @classmethod
    def is_func_actor_attachment_supported(cls, actor : "FuncActor") -> bool:
        cls._build_func_actor_mixin_properties()
        for prop in cls._func_actor_mixin_properties:
            if not actor.has_wrapper_attr(prop):
                return False
        return True
    
    @classmethod
    def _build_func_actor_mixin_properties(cls) -> None:
        if not hasattr(cls, "_func_actor_mixin_properties"):
            properties_func_actor = vars(cls.mixin_func_actor_type).keys() if cls.mixin_func_actor_type is not None else []
            properties_func_actor = [p for p in properties_func_actor if not p.startswith("_")]
            cls._func_actor_mixin_properties = properties_func_actor

class Actor(ABC, Generic[BDeviceType, BDtypeType, BRNGType]):
    """
    Actor Interface

    Note that each sensor attached should have control_timestep equal to the actor's control_timestep, or the control_timestep should be dividends of the actor's control_timestep.
    """
    backend : ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType]

    extra_observation_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]] = None
    extra_action_space : Optional[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]] = None

    control_timestep : float
    is_real : bool

    @property
    def action_space(self) -> Union[Space[Any, Any, BDeviceType, BDtypeType, BRNGType], DictSpace[BDeviceType, BDtypeType, BRNGType]]:
        if len(self._mixin_action_spaces) > 0:
            if self.extra_action_space is None:
                new_spaces = self._mixin_action_spaces
            else:
                if isinstance(self.extra_action_space, DictSpace):
                    new_spaces = {
                        **self._mixin_action_spaces,
                        **self.extra_action_space.spaces
                    }
                else:
                    new_spaces = {
                        **self._mixin_action_spaces,
                        'extra': self.extra_action_space
                    }
            return DictSpace(
                backend=self.backend,
                spaces=new_spaces,
                device=self.device
            )
        else:
            assert self.extra_action_space is not None
            return self.extra_action_space
    
    @property
    def onboard_observation_space(self) -> DictSpace[BDeviceType, BDtypeType, BRNGType]:
        if len(self._mixin_observation_spaces) > 0:
            if self.extra_observation_space is None:
                new_spaces = self._mixin_observation_spaces
            else:
                new_spaces = {
                    **self._mixin_observation_spaces,
                    **self.extra_observation_space.spaces
                }
            return DictSpace(
                backend=self.backend,
                spaces=new_spaces,
                device=self.device
            )
        else:
            assert self.extra_observation_space is not None
            return self.extra_observation_space

    @property
    def observation_space(self) -> DictSpace[BDeviceType, BDtypeType, BRNGType]:
        space = self.onboard_observation_space.spaces.copy()
        for key, sensor in self.sensors.items():
            space[key] = sensor.observation_space
        return DictSpace(
            backend=self.backend,
            spaces=space,
            device=self.device
        )

    def __init__(
        self,
        mixins : Iterable[ActorMixin] = [],
    ):
        self._sensors : Dict[str, Sensor] = {}
        self._mixins : List[ActorMixin] = list(mixins)
        self._update_mixins_internal()
    
    def implements_mixin(self, mixin : Type[ActorMixin]) -> bool:
        return any(m is mixin or isinstance(m, mixin) for m in self.mixins)

    def as_mixin(self, mixin : Type[ActorMixin[ActorMixinT, Any]]) -> ActorMixinT:
        if not self.implements_mixin(mixin):
            raise ValueError(f"Actor does not implement mixin {mixin.mixin_name}")
        return self

    def _update_mixins_internal(self) -> None:
        mixin_action_spaces = {}
        mixin_observation_spaces = {}

        for mixin in self.mixins:
            action_space = mixin.get_mixin_action_space(self, self.backend, self.device)
            if action_space is not None:
                mixin_action_spaces[mixin.mixin_name] = action_space
            observation_space = mixin.get_mixin_observation_space(self, self.backend, self.device)
            if observation_space is not None:
                mixin_observation_spaces[mixin.mixin_name] = observation_space
        self._mixin_action_spaces = mixin_action_spaces
        self._mixin_observation_spaces = mixin_observation_spaces
    
    def update(self, last_step_elapsed : float) -> None:
        self.update_onboard(last_step_elapsed=last_step_elapsed)
        self.update_sensors(last_step_elapsed=last_step_elapsed)

    def get_data(self) -> Dict[str, Any]:
        onboard_data = self.get_onboard_data()
        sensors_data = self.get_sensors_data()
        onboard_data.update(sensors_data)
        return onboard_data

    @abstractmethod
    def update_onboard(self, last_step_elapsed : float) -> None:
        """
        Update the onboard observations with a new state (e.g. from the environment).
        """
        raise NotImplementedError

    def get_extra_data(self) -> Dict[str, Any]:
        """
        Get the extra data from the actor.
        """
        raise NotImplementedError

    def get_onboard_data(self) -> Dict[str, Any]:
        if len(self._mixin_observation_spaces) == 0:
            return self.get_extra_data()
        
        mixin_datas = {}
        for mixin in self.mixins:
            if mixin.mixin_name in self._mixin_observation_spaces.keys():
                mixin_datas[mixin.mixin_name] = mixin.read_mixin_data(self)
        if self.extra_observation_space is not None:
            if isinstance(self.extra_observation_space, DictSpace):
                obs_data = {
                    **mixin_datas,
                    **self.get_extra_data()
                }
            else:
                obs_data = {
                    **mixin_datas,
                    'extra': self.get_extra_data()
                }
        else:
            obs_data = mixin_datas
        return obs_data
    
    @property
    def mixins(self) -> List[ActorMixin]:
        return self._mixins

    @property
    def sensors(self) -> Dict[str, Sensor]:
        return self._sensors

    def update_sensors(self, last_step_elapsed : float) -> None:
        for sensor in self.sensors.values():
            sensor.update(last_step_elapsed=last_step_elapsed)
    
    def get_sensors_data(self) -> Dict[str, Any]:
        ret = {}
        for key, sensor in self.sensors.items():
            ret[key] = sensor.get_data()
        return ret

    def set_next_extra_action(self, action: Any) -> None:
        """
        Sets the next action to be executed by the actor when the environment steps.
        This method should be called before the environment steps, and should be non-blocking.
        """
        raise NotImplementedError

    def set_next_action(self, action: Any) -> None:
        if len(self._mixin_action_spaces) > 0:
            assert isinstance(action, dict)
            action : Dict[str, Any] = action.copy()
            for mixin in self.mixins:
                if mixin.mixin_name in self._mixin_action_spaces.keys():
                    mixin.apply_mixin_action(self, action.pop(mixin.mixin_name))
            if self.extra_action_space is not None:
                extra_action = action.pop('extra') if 'extra' in action.keys() else action
                self.set_next_extra_action(extra_action)
        else:
            self.set_next_extra_action(action)
    
    def pre_environment_step(self, last_step_elapsed : float) -> None:
        pass

    def reset(self) -> None:
        for sensor in self.sensors.values():
            sensor.reset()
    
    def close(self) -> None:
        for sensor in self.sensors.values():
            sensor.close()
        self._sensors.clear()

    def __del__(self) -> None:
        self.close()

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Actor":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["Actor"]:
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

ActorStateT = TypeVar("ActorStateT", covariant=True)

@dataclass(frozen=True)
class FuncActorCombinedState(Generic[ActorStateT]):
    actor_state : ActorStateT
    sensor_states : Dict[str, Any]


class FuncActor(
    ABC,
    Generic[StateType, ActorStateT, BDeviceType, BDtypeType, BRNGType]
):
    """
    FuncActor Interface

    Note that each sensor attached should have control_timestep equal to the actor's control_timestep, or the control_timestep should be dividends of the actor's control_timestep.
    """
    backend : ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType]

    extra_observation_space : Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]] = None
    extra_action_space : Optional[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]] = None
    
    control_timestep : float
    is_real : bool

    def __init__(
        self,
        mixins : Iterable[ActorMixin] = [],
    ):
        self._sensors : Dict[
            str, FuncSensor[Any, Any, Any, BDeviceType, BDtypeType, BRNGType]
        ] = {}
        self._mixins : List[ActorMixin] = list(mixins)
        self._update_mixins_internal()

    def _update_mixins_internal(self) -> None:
        mixin_action_spaces = {}
        mixin_observation_spaces = {}

        for mixin in self.mixins:
            action_space = mixin.get_mixin_action_space(self, self.backend, self.device)
            if action_space is not None:
                mixin_action_spaces[mixin.mixin_name] = action_space
            observation_space = mixin.get_mixin_observation_space(self, self.backend, self.device)
            if observation_space is not None:
                mixin_observation_spaces[mixin.mixin_name] = observation_space
        self._mixin_action_spaces = mixin_action_spaces
        self._mixin_observation_spaces = mixin_observation_spaces

    def implements_mixin(self, mixin : Type[ActorMixin]) -> bool:
        return any(m is mixin or isinstance(m, mixin) for m in self.mixins)

    def as_mixin(self, mixin : Type[ActorMixin[Any, ActorMixinFuncT]]) -> ActorMixinFuncT:
        if not self.implements_mixin(mixin):
            raise ValueError(f"Actor does not implement mixin {mixin.mixin_name}")
        return self

    @property
    def sensors(self) -> Dict[str, FuncSensor[Any, Any, Any, BDeviceType, BDtypeType, BRNGType]]:
        return self._sensors

    @property
    def mixins(self) -> List[ActorMixin]:
        return self._mixins

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.onboard_observation_space.device
    
    @property
    def backend(self) -> ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]:
        return self.onboard_observation_space.backend
    
    @property
    def action_space(self) -> Union[Space[Any, Any, BDeviceType, BDtypeType, BRNGType], DictSpace[BDeviceType, BDtypeType, BRNGType]]:
        if len(self._mixin_action_spaces) > 0:
            if self.extra_action_space is None:
                new_spaces = self._mixin_action_spaces
            else:
                if isinstance(self.extra_action_space, DictSpace):
                    new_spaces = {
                        **self._mixin_action_spaces,
                        **self.extra_action_space.spaces
                    }
                else:
                    new_spaces = {
                        **self._mixin_action_spaces,
                        'extra': self.extra_action_space
                    }
            return DictSpace(
                backend=self.backend,
                spaces=new_spaces,
                device=self.device
            )
        else:
            assert self.extra_action_space is not None
            return self.extra_action_space
    
    @property
    def onboard_observation_space(self) -> DictSpace[BDeviceType, BDtypeType, BRNGType]:
        if len(self._mixin_observation_spaces) > 0:
            if self.extra_observation_space is None:
                new_spaces = self._mixin_observation_spaces
            else:
                new_spaces = {
                    **self._mixin_observation_spaces,
                    **self.extra_observation_space.spaces
                }
            return DictSpace(
                backend=self.backend,
                spaces=new_spaces,
                device=self.device
            )
        else:
            assert self.extra_observation_space is not None
            return self.extra_observation_space

    @property
    def observation_space(self) -> DictSpace[BDeviceType, BDtypeType, BRNGType]:
        space = self.onboard_observation_space.spaces.copy()
        for key, sensor in self.sensors.items():
            space[key] = sensor.observation_space
        return DictSpace(
            backend=self.backend,
            spaces=space,
            device=self.device
        )

    @abstractmethod
    def onboard_initial(
        self,
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        *,
        seed : int
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def onboard_reset(
        self,
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : ActorStateT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def onboard_step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : ActorStateT,
        last_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        raise NotImplementedError

    def get_data_extra(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : ActorStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT,
        Optional[Dict[str, Any]]
    ]:
        raise NotImplementedError

    def get_data_onboard(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : ActorStateT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT,
        Dict[str, Any]
    ]:
        if len(self._mixin_observation_spaces) == 0:
            return self.get_data_extra(
                state,
                common_state,
                actor_state,
                last_control_step_elapsed
            )

        mixin_datas = {}
        for mixin in self.mixins:
            if mixin.mixin_name in self._mixin_observation_spaces.keys():
                state, common_state, actor_state, mixin_datas[mixin.mixin_name] = mixin.read_mixin_data_func(
                    self, 
                    state, 
                    common_state, 
                    actor_state, 
                    last_control_step_elapsed
                )
        if self.extra_observation_space is not None:
            state, common_state, actor_state, extra_data = self.get_data_extra(
                state,
                common_state,
                actor_state,
                last_control_step_elapsed
            )
            if isinstance(self.extra_observation_space, DictSpace):
                obs_data = {
                    **mixin_datas,
                    **extra_data
                }
            else:
                obs_data = {
                    **mixin_datas,
                    'extra': extra_data
                }
        else:
            obs_data = mixin_datas
        return state, common_state, actor_state, obs_data
        

    def set_next_extra_action(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : ActorStateT,
        action : Any,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        raise NotImplementedError

    def set_next_action(
        self, 
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : ActorStateT,
        action : Any,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        if len(self._mixin_action_spaces) > 0:
            assert isinstance(action, dict)
            action : Dict[str, Any] = action.copy()
            for mixin in self.mixins:
                if mixin.mixin_name in self._mixin_action_spaces.keys():
                    state, common_state, actor_state = mixin.apply_mixin_action_func(
                        self,
                        state,
                        common_state,
                        actor_state,
                        action.pop(mixin.mixin_name),
                        last_control_step_elapsed
                    )
            if self.extra_action_space is not None:
                extra_action = action.pop('extra') if 'extra' in action.keys() else action
                state, common_state, actor_state = self.set_next_extra_action(
                    state,
                    common_state,
                    actor_state,
                    extra_action,
                    last_control_step_elapsed
                )
        else:
            state, common_state, actor_state = self.set_next_extra_action(
                state,
                common_state,
                actor_state,
                action,
                last_control_step_elapsed
            )
        return state, common_state, actor_state
    
    @abstractmethod
    def onboard_close(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : ActorStateT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType]
    ]:
        raise NotImplementedError
    
    def initial(
        self,
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        *onboard_args,
        seed : int,
        sensor_kwargs : Optional[Dict[str, Dict[str, Any]]] = None,
        **onboard_kwargs
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        FuncActorCombinedState[ActorStateT]
    ]:
        state, common_state, actor_state = self.onboard_initial(
            world,
            state,
            common_state,
            *onboard_args,
            seed=seed,
            **onboard_kwargs
        )
        
        sensor_states = {}
        for key, sensor in self.sensors.items():
            current_sensor_kwargs = sensor_kwargs[key] if sensor_kwargs is not None and key in sensor_kwargs.keys() else {}
            state, common_state, sensor_state = sensor.initial(
                world,
                state,
                common_state,
                seed=seed,
                **current_sensor_kwargs
            )
            sensor_states[key] = sensor_state
        
        return (
            state,
            common_state,
            FuncActorCombinedState(
                actor_state=actor_state,
                sensor_states=sensor_states
            )
        )
    
    def reset(
        self,
        world : FuncWorld[StateType, BDeviceType, BDtypeType, BRNGType],
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        combined_state : FuncActorCombinedState[ActorStateT],
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        FuncActorCombinedState[ActorStateT]
    ]:
        state, common_state, actor_state = self.onboard_reset(
            world,
            state,
            common_state,
            actor_state=combined_state.actor_state
        )
        
        sensor_states = {}
        for key, sensor in self.sensors.items():
            state, common_state, sensor_states[key] = sensor.reset(
                world,
                state,
                common_state,
                sensor_state=combined_state.sensor_states[key]
            )
        
        return (
            state,
            common_state,
            FuncActorCombinedState(
                actor_state=actor_state, 
                sensor_states=sensor_states
            )
        )

    def step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        combined_state : FuncActorCombinedState[ActorStateT],
        last_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        FuncActorCombinedState[ActorStateT]
    ]:
        state, common_state, actor_state = self.onboard_step(
            state=state,
            common_state=common_state,
            actor_state=combined_state.actor_state,
            last_step_elapsed=last_step_elapsed
        )
        
        sensor_states = {}
        for key, sensor in self.sensors.items():
            state, common_state, sensor_states[key] = sensor.step(
                state=state,
                common_state=common_state,
                sensor_state=combined_state.sensor_states[key],
                last_step_elapsed=last_step_elapsed
            )
        
        return (
            state,
            common_state,
            FuncActorCombinedState(
                actor_state=actor_state, 
                sensor_states=sensor_states
            )
        )
    
    def get_data(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        combined_state : FuncActorCombinedState[ActorStateT],
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        FuncActorCombinedState[ActorStateT],
        Dict[str, Any]
    ]:
        state, common_state, actor_state, onboard_data = self.get_data_onboard(
            state,
            common_state,
            actor_state=combined_state.actor_state,
            last_control_step_elapsed=last_control_step_elapsed
        )
        
        sensors_state = {}
        sensors_data = {}
        for key, sensor in self.sensors.items():
            state, common_state, sensors_state[key], sensors_data[key] = sensor.get_data(
                state,
                common_state,
                sensor_state=combined_state.sensor_states[key],
                last_control_step_elapsed=last_control_step_elapsed
            )
        
        onboard_data.update(sensors_data)
        return state, common_state, FuncActorCombinedState(
            actor_state=actor_state,
            sensor_states=sensors_state
        ), onboard_data
    
    def close(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        combined_state : FuncActorCombinedState[ActorStateT]
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType]
    ]:
        state, common_state = self.onboard_close(
            state,
            common_state,
            actor_state=combined_state.actor_state
        )
        
        for key, sensor in self.sensors.items():
            state, common_state = sensor.close(
                state,
                common_state,
                sensor_state=combined_state.sensor_states[key]
            )
        
        return state, common_state
    
    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "Actor":
        return self
    
    @property
    def prev_wrapper_layer(self) -> Optional["Actor"]:
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

ActorWrapperBDeviceType = TypeVar("ActorWrapperBDeviceType")
ActorWrapperBDtypeType = TypeVar("ActorWrapperBDtypeType")
ActorWrapperBRNGType = TypeVar("ActorWrapperBRNGType")

class ActorWrapper(
    Generic[
        ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType,
        BDeviceType, BDtypeType, BRNGType
    ],
    Actor[ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]
):
    def __init__(
        self,
        actor : Actor[BDeviceType, BDtypeType, BRNGType]
    ):
        self.actor = actor
        self.extra_action_space : Space[
            Any, Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType
        ] = actor.extra_action_space
        self.extra_observation_space : Optional[DictSpace[
            ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType
        ]] = actor.extra_observation_space
        self._sensors : Optional[Dict[str, Sensor]] = None
        self._mixins : Optional[List[ActorMixin]] = None
        self._mixin_action_spaces = actor._mixin_action_spaces
        self._mixin_observation_spaces = actor._mixin_observation_spaces

    @property
    def control_timestep(self) -> float:
        return self.actor.control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value : float) -> None:
        self.actor.control_timestep = value

    @property
    def is_real(self) -> bool:
        return self.actor.is_real
    
    def update_onboard(self, last_step_elapsed : float) -> None:
        self.actor.update_onboard(last_step_elapsed=last_step_elapsed)
    
    def get_extra_data(self) -> Dict[str, Any]:
        return self.actor.get_extra_data()
    
    @property
    def sensors(self) -> Dict[str, Sensor]:
        return self._sensors or self.actor.sensors
    
    @sensors.setter
    def sensors(self, value : Dict[str, Sensor]) -> None:
        self._sensors = value

    @property
    def mixins(self) -> List[ActorMixin]:
        return self._mixins or self.actor.mixins
    
    @mixins.setter
    def mixins(self, value : List[ActorMixin]) -> None:
        self._mixins = value
        self._update_mixins_internal()
    
    def set_next_extra_action(self, action: Any) -> None:
        return self.actor.set_next_extra_action(action)

    def pre_environment_step(self, last_step_elapsed : float) -> None:
        self.actor.pre_environment_step(last_step_elapsed=last_step_elapsed)

    def reset(self) -> None:
        self.actor.reset()
    
    def close(self) -> None:
        super().close()
        self.actor.close()
    
    # ========== Property Override for Mixins ==========
    def __getattr__(self, name : str) -> Any:
        if any(mixin.is_actor_mixin_property(name) for mixin in self.mixins):
            return self.get_wrapper_attr(name)
        else:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )
    
    # ========== Wrapper methods ==========

    @property
    def unwrapped(self) -> "Actor":
        return self.actor.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "Actor[BDeviceType, BDtypeType, BRNGType]":
        return self.actor
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.actor.has_wrapper_attr(name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.actor.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        sub_actor = self
        attr_set = False

        while attr_set is False and sub_actor is not None:
            if hasattr(sub_actor, name):
                setattr(sub_actor, name, value)
                attr_set = True
            else:
                sub_actor = sub_actor.prev_wrapper_layer

        if attr_set is False and sub_actor is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )

ActorWrapperStateT = TypeVar("ActorWrapperStateT")
class FuncActorWrapper(
    Generic[
        StateType, 
        ActorWrapperStateT, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType,
        ActorStateT, BDeviceType, BDtypeType, BRNGType    
    ],
    FuncActor[StateType, ActorWrapperStateT, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]
):
    def __init__(
        self,
        actor : FuncActor[StateType, ActorStateT, BDeviceType, BDtypeType, BRNGType]
    ):
        self.actor = actor
        self.extra_action_space : Optional[Space[
            Any, Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType
        ]] = actor.extra_action_space
        self.extra_observation_space : Optional[DictSpace[
            ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType
        ]] = actor.extra_observation_space
        self._sensors : Optional[Dict[str, FuncSensor[Any, Any, Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]]] = None
        self._mixins : Optional[List[ActorMixin]] = None
        self._mixin_action_spaces = actor._mixin_action_spaces
        self._mixin_observation_spaces = actor._mixin_observation_spaces
    
    def _translate_to_inner_actor_state(
        self, 
        state : ActorWrapperStateT,
    ) -> ActorStateT:
        raise NotImplementedError
    
    def _translate_to_outer_actor_state(
        self, 
        state : ActorStateT,
        old_wrapper_state : ActorWrapperStateT,
    ) -> ActorWrapperStateT:
        raise NotImplementedError
    
    @property
    def control_timestep(self) -> float:
        return self.actor.control_timestep
    
    @property
    def is_real(self) -> bool:
        return self.actor.is_real
    
    @property
    def mixins(self) -> List[ActorMixin]:
        return self._mixins or self.actor.mixins
    
    @mixins.setter
    def mixins(self, value : List[ActorMixin]) -> None:
        self._mixins = value
        self._update_mixins_internal()

    @property
    def sensors(self) -> Dict[str, FuncSensor[Any, Any, Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]]:
        return self._sensors or self.actor.sensors
    
    @sensors.setter
    def sensors(self, value : Dict[str, FuncSensor[Any, Any, Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]]) -> None:
        self._sensors = value
    
    @property
    def device(self) -> Optional[ActorWrapperBDeviceType]:
        return self.actor.device
    
    @property
    def backend(self) -> ComputeBackend[Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]:
        return self.actor.backend
    
    @property
    def observation_space(self) -> DictSpace[ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]:
        return self.actor.observation_space
    
    def onboard_initial(
        self, 
        world : FuncWorld[StateType, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType],
        state: StateType, 
        common_state: FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        *args, 
        seed: int,
        **kwargs
    ) -> Tuple[
        StateType, 
        FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        ActorWrapperStateT
    ]:
        return self.actor.onboard_initial(
            world,
            state,
            common_state,
            *args,
            seed=seed,
            **kwargs
        )
    
    def onboard_reset(
        self, 
        world : FuncWorld[StateType, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType],
        state: StateType, 
        common_state: FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        actor_state: ActorWrapperStateT
    ) -> Tuple[
        StateType, 
        FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        ActorWrapperStateT
    ]:
        state, common_state, inner_actor_state = self.actor.onboard_reset(
            world,
            state,
            common_state, 
            actor_state=self._translate_to_inner_actor_state(actor_state)
        )
        return state, common_state, self._translate_to_outer_actor_state(
            inner_actor_state, 
            actor_state
        )

    def onboard_step(
        self, 
        state: StateType, 
        common_state: FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        actor_state: ActorWrapperStateT, 
        last_step_elapsed: float
    ) -> Tuple[
        StateType, 
        FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        ActorWrapperStateT
    ]:
        state, common_state, inner_actor_state = self.actor.onboard_step(
            state, 
            common_state, 
            actor_state=self._translate_to_inner_actor_state(actor_state), 
            last_step_elapsed=last_step_elapsed
        )
        return state, common_state, self._translate_to_outer_actor_state(
            inner_actor_state, 
            actor_state
        )

    def get_data_extra(
        self, 
        state: StateType, 
        common_state: FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        actor_state: ActorWrapperStateT,
        last_control_step_elapsed: float
    ) -> Tuple[
        StateType, 
        FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        ActorWrapperStateT, 
        Dict[str, Any]
    ]:
        state, common_state, inner_actor_state, onboard_data = self.actor.get_data_extra(
            state, 
            common_state, 
            actor_state=self._translate_to_inner_actor_state(actor_state), 
            last_control_step_elapsed=last_control_step_elapsed
        )
        return state, common_state, self._translate_to_outer_actor_state(
            inner_actor_state, 
            actor_state
        ), onboard_data

    def set_next_extra_action(
        self, 
        state: StateType, 
        common_state: FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        actor_state: ActorWrapperStateT, 
        action: Any,
        last_control_step_elapsed: float
    ) -> Tuple[
        StateType, 
        FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        ActorWrapperStateT
    ]:
        state, common_state, inner_actor_state = self.actor.set_next_action(
            state, 
            common_state, 
            actor_state=self._translate_to_inner_actor_state(actor_state), 
            action=action, 
            last_control_step_elapsed=last_control_step_elapsed
        )
        return state, common_state, self._translate_to_outer_actor_state(
            inner_actor_state, 
            actor_state
        )
    
    def onboard_close(
        self, 
        state: StateType, 
        common_state: FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        actor_state: ActorWrapperStateT
    ) -> Tuple[
        StateType, 
        FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType]
    ]:
        return self.actor.onboard_close(
            state, 
            common_state, 
            actor_state=self._translate_to_inner_actor_state(actor_state)
        )

    # ========== Property Override for Mixins ==========
    def __getattr__(self, name : str) -> Any:
        if any(mixin.is_func_actor_mixin_property(name) for mixin in self.mixins):
            return self.get_wrapper_attr(name)
        else:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )

    # ========== Wrapper methods ==========
    @property
    def unwrapped(self) -> "FuncActor":
        return self.actor.unwrapped
    
    @property
    def prev_wrapper_layer(self) -> "FuncActor[StateType, ActorStateT, Any, BDeviceType, BDtypeType, BRNGType]":
        return self.actor
    
    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.actor.has_wrapper_attr(name)
    
    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.actor.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {type(self).__name__} has no attribute {name!r}"
                ) from e

    def set_wrapper_attr(self, name: str, value: Any):
        sub_actor = self
        attr_set = False

        while attr_set is False and sub_actor is not None:
            if hasattr(sub_actor, name):
                setattr(sub_actor, name, value)
                attr_set = True
            else:
                sub_actor = sub_actor.prev_wrapper_layer

        if attr_set is False and sub_actor is None:
            raise AttributeError(
                f"wrapper {type(self).__name__} has no attribute {name!r}"
            )