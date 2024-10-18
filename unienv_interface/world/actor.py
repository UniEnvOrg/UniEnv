from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from ..space import Space
from ..space.dict import Dict as DictSpace
from ..backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from ..env_base.funcenv import FuncEnvCommonState, FuncEnv, StateType
from .sensor import Sensor, FuncSensor
from dataclasses import dataclass

ActorActT = TypeVar("ActorActT", covariant=True)

"""
Actor Interface

Note that each sensor attached should have control_timestep equal to the actor's control_timestep, or the control_timestep should be dividends of the actor's control_timestep.
"""
class Actor(ABC, Generic[ActorActT, BDeviceType, BDtypeType, BRNGType]):
    onboard_observation_space : DictSpace[BDeviceType, BDtypeType, BRNGType]
    action_space : Space[ActorActT, Any, BDeviceType, BDtypeType, BRNGType]
    control_timestep : float
    is_real : bool

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
        self
    ):
        self._sensors : Dict[str, Sensor] = {}

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.observation_space.device
    
    @property
    def backend(self) -> Type[ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]]:
        return self.observation_space.backend

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

    @abstractmethod
    def get_onboard_data(self) -> Dict[str, Any]:
        """Get the data from the sensor."""
        raise NotImplementedError
    
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

    @abstractmethod
    def set_next_action(self, action: ActorActT) -> None:
        """
        Sets the next action to be executed by the actor when the environment steps.
        This method should be called before the environment steps, and should be non-blocking.
        """
        raise NotImplementedError
    
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

ActorStateT = TypeVar("ActorStateT")

@dataclass(frozen=True)
class FuncActorCombinedState(Generic[ActorStateT]):
    actor_state : ActorStateT
    sensor_states : Dict[str, Any]

"""
FuncActor Interface

Note that each sensor attached should have control_timestep equal to the actor's control_timestep, or the control_timestep should be dividends of the actor's control_timestep.
"""
class FuncActor(
    ABC,
    Generic[StateType, ActorStateT, ActorActT, BDeviceType, BDtypeType, BRNGType]
):
    onboard_observation_space : DictSpace[BDeviceType, BDtypeType, BRNGType]
    action_space : Space[ActorActT, Any, BDeviceType, BDtypeType, BRNGType]
    control_timestep : float
    is_real : bool

    def __init__(
        self
    ):
        self._sensors : Dict[
            str, FuncSensor[Any, Any, Any, BDeviceType, BDtypeType, BRNGType]
        ] = {}

    @property
    def sensors(self) -> Dict[str, FuncSensor[Any, Any, Any, BDeviceType, BDtypeType, BRNGType]]:
        return self._sensors

    @property
    def device(self) -> Optional[BDeviceType]:
        return self.onboard_observation_space.device
    
    @property
    def backend(self) -> Type[ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]]:
        return self.onboard_observation_space.backend
    
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

    @abstractmethod
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
        """
        Reads the data when the actor is actionable.
        """
        raise NotImplementedError

    @abstractmethod
    def set_next_action(
        self, 
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : ActorStateT,
        action : ActorActT,
        last_control_step_elapsed : float
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT
    ]:
        raise NotImplementedError
    
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
            state=state,
            common_state=common_state,
            *onboard_args,
            seed=seed,
            **onboard_kwargs
        )
        
        sensor_states = {}
        for key, sensor in self.sensors.items():
            current_sensor_kwargs = sensor_kwargs[key] if sensor_kwargs is not None and key in sensor_kwargs.keys() else {}
            state, common_state, sensor_state = sensor.initial(
                state=state,
                common_state=common_state,
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
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        combined_state : FuncActorCombinedState[ActorStateT],
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        FuncActorCombinedState[ActorStateT]
    ]:
        state, common_state, actor_state = self.onboard_reset(
            state=state,
            common_state=common_state,
            actor_state=combined_state.actor_state
        )
        
        sensor_states = {}
        for key, sensor in self.sensors.items():
            state, common_state, sensor_state = sensor.reset(
                state=state,
                common_state=common_state,
                sensor_state=combined_state.sensor_states[key]
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
            state, common_state, sensor_state = sensor.step(
                state=state,
                common_state=common_state,
                sensor_state=combined_state.sensor_states[key],
                last_step_elapsed=last_step_elapsed
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
            state=state,
            common_state=common_state,
            actor_state=combined_state.actor_state,
            last_control_step_elapsed=last_control_step_elapsed
        )
        
        sensors_state = {}
        sensors_data = {}
        for key, sensor in self.sensors.items():
            state, common_state, sensors_state[key], sensors_data[key] = sensor.get_data(
                state=state,
                common_state=common_state,
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
            state=state,
            common_state=common_state,
            actor_state=combined_state.actor_state
        )
        
        for key, sensor in self.sensors.items():
            state, common_state = sensor.close(
                state=state,
                common_state=common_state,
                sensor_state=combined_state.sensor_states[key]
            )
        
        return state, common_state

ActorWrapperActT = TypeVar("ActorWrapperActT")
ActorWrapperStateT = TypeVar("ActorWrapperStateT")
ActorWrapperBDeviceType = TypeVar("ActorWrapperBDeviceType")
ActorWrapperBDtypeType = TypeVar("ActorWrapperBDtypeType")
ActorWrapperBRNGType = TypeVar("ActorWrapperBRNGType")
class FuncActorWrapper(
    Generic[
        StateType, ActorWrapperStateT, ActorWrapperActT, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType,
        ActorStateT, ActorActT, BDeviceType, BDtypeType, BRNGType    
    ],
    FuncActor[StateType, ActorWrapperStateT, ActorWrapperStateT, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]
):
    def __init__(
        self,
        actor : FuncActor[StateType, ActorStateT, ActorActT, BDeviceType, BDtypeType, BRNGType]
    ):
        self.actor = actor
    
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
    def onboard_observation_space(self) -> DictSpace[ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]:
        return self.actor.onboard_observation_space
    
    @property
    def action_space(self) -> Space[ActorWrapperActT, Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]:
        return self.actor.action_space
    
    @property
    def control_timestep(self) -> float:
        return self.actor.control_timestep
    
    @property
    def is_real(self) -> bool:
        return self.actor.is_real
    
    @property
    def sensors(self) -> Dict[str, FuncSensor[Any, Any, Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]]:
        return self.actor.sensors
    
    @property
    def device(self) -> Optional[ActorWrapperBDeviceType]:
        return self.actor.device
    
    @property
    def backend(self) -> Type[ComputeBackend[Any, ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]]:
        return self.actor.backend
    
    @property
    def observation_space(self) -> DictSpace[ActorWrapperBDeviceType, ActorWrapperBDtypeType, ActorWrapperBRNGType]:
        return self.actor.observation_space
    
    def onboard_initial(
        self, 
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
            state=state, 
            common_state=common_state, 
            *args,
            seed=seed,
            **kwargs
        )
    
    def onboard_reset(
        self, 
        state: StateType, 
        common_state: FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        actor_state: ActorWrapperStateT
    ) -> Tuple[
        StateType, 
        FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        ActorWrapperStateT
    ]:
        state, common_state, inner_actor_state = self.actor.onboard_reset(
            state=state, 
            common_state=common_state, 
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
            state=state, 
            common_state=common_state, 
            actor_state=self._translate_to_inner_actor_state(actor_state), 
            last_step_elapsed=last_step_elapsed
        )
        return state, common_state, self._translate_to_outer_actor_state(
            inner_actor_state, 
            actor_state
        )

    def get_data_onboard(
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
        state, common_state, inner_actor_state, onboard_data = self.actor.get_data_onboard(
            state=state, 
            common_state=common_state, 
            actor_state=self._translate_to_inner_actor_state(actor_state), 
            last_control_step_elapsed=last_control_step_elapsed
        )
        return state, common_state, self._translate_to_outer_actor_state(
            inner_actor_state, 
            actor_state
        ), onboard_data
    
    def set_next_action(
        self, 
        state: StateType, 
        common_state: FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        actor_state: ActorWrapperStateT, 
        action: ActorWrapperActT,
        last_control_step_elapsed: float
    ) -> Tuple[
        StateType, 
        FuncEnvCommonState[ActorWrapperBDeviceType, ActorWrapperBRNGType], 
        ActorWrapperStateT
    ]:
        state, common_state, inner_actor_state = self.actor.set_next_action(
            state=state, 
            common_state=common_state, 
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
            state=state, 
            common_state=common_state, 
            actor_state=self._translate_to_inner_actor_state(actor_state)
        )
