from typing import Generic, Any, TypeVar, Optional, Dict, Tuple, Sequence, List, Type
from abc import ABC, abstractmethod
from ..space import Space
from ..space.dict import Dict as DictSpace
from ..backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from ..env_base.funcenv import FuncEnvCommonState, FuncEnv, StateType
from .sensor import Sensor, FuncSensor
from dataclasses import dataclass

ActorActT = TypeVar("ActorActT", covariant=True)
class Actor(ABC, Generic[ActorActT, BDeviceType, BDtypeType, BRNGType]):
    onboard_observation_space : DictSpace[BDeviceType, BDtypeType, BRNGType]
    action_space : Space[ActorActT, Any, BDeviceType, BDtypeType, BRNGType]
    control_timestep : float
    is_real : bool

    @property
    def observation_space(self) -> DictSpace[BDeviceType, BDtypeType, BRNGType]:
        space = self.onboard_observation_space.spaces.copy()
        for key, sensor in self._sensors.items():
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

    def update(self) -> None:
        self.update_onboard()
        self.update_sensors()
    
    def get_data(self) -> Optional[Dict[str, Any]]:
        onboard_data = self.get_onboard_data()
        sensors_data = self.get_sensors_data()
        if onboard_data is None or sensors_data is None:
            return None
        onboard_data.update(sensors_data)
        return onboard_data

    @abstractmethod
    def update_onboard(self) -> None:
        """
        Update the onboard observations with a new state (e.g. from the environment).
        """
        raise NotImplementedError

    @abstractmethod
    def get_onboard_data(self) -> Optional[Dict[str, Any]]:
        """Get the data from the sensor."""
        raise NotImplementedError
    
    @property
    def sensors(self) -> Dict[str, Sensor]:
        return self._sensors

    def update_sensors(self) -> None:
        for sensor in self._sensors.values():
            sensor.update()
    
    def get_sensors_data(self) -> Optional[Dict[str, Any]]:
        ret = {}
        for key, sensor in self._sensors.items():
            dat = sensor.get_data()
            if dat is not None:
                ret[key] = dat
            else:
                return None
        return ret

    @abstractmethod
    def set_next_action(self, action: ActorActT) -> None:
        """
        Sets the next action to be executed by the actor when the environment steps.
        This method should be called before the environment steps, and should be non-blocking.
        """
        raise NotImplementedError

    def reset(self) -> None:
        for sensor in self._sensors.values():
            sensor.reset()
    
    def close(self) -> None:
        for sensor in self._sensors.values():
            sensor.close()
        self._sensors.clear()

    def __del__(self) -> None:
        self.close()


ActorStateT = TypeVar("ActorStateT")

@dataclass
class FuncActorCombinedState(Generic[ActorStateT]):
    actor_state : ActorStateT
    sensor_states : Dict[str, Any]

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
    def device(self) -> Optional[BDeviceType]:
        return self.observation_space.device
    
    @property
    def backend(self) -> Type[ComputeBackend[Any, BDeviceType, BDtypeType, BRNGType]]:
        return self.observation_space.backend
    
    @property
    def observation_space(self) -> DictSpace[BDeviceType, BDtypeType, BRNGType]:
        space = self.onboard_observation_space.spaces.copy()
        for key, sensor in self._sensors.items():
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
        ActorStateT,
        Dict[str, Any] # Onboard Observation
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
        ActorStateT,
        Dict[str, Any] # Onboard Observation
    ]:
        raise NotImplementedError
    
    @abstractmethod
    def onboard_step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        actor_state : ActorStateT,
        action : ActorActT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        ActorStateT,
        Dict[str, Any], # Onboard Observation
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
        FuncActorCombinedState[ActorStateT],
        Dict[str, Any] # All observations
    ]:
        state, common_state, actor_state, onboard_obs = self.onboard_initial(
            state=state,
            common_state=common_state,
            *onboard_args,
            seed=seed,
            **onboard_kwargs
        )
        
        sensor_states = {}
        sensor_obss = {}
        for key, sensor in self._sensors.items():
            current_sensor_kwargs = sensor_kwargs[key] if sensor_kwargs is not None and key in sensor_kwargs.keys() else {}
            state, common_state, sensor_state, sensor_obs = sensor.initial(
                state=state,
                common_state=common_state,
                seed=seed,
                **current_sensor_kwargs
            )
            sensor_states[key] = sensor_state
            sensor_obss[key] = sensor_obs
        
        combined_obs = onboard_obs.copy()
        combined_obs.update(sensor_obss)
        return (
            state,
            common_state,
            FuncActorCombinedState(
                actor_state=actor_state, 
                sensor_states=sensor_states
            ),
            combined_obs
        )
    
    def reset(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        combined_state : FuncActorCombinedState[ActorStateT],
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        FuncActorCombinedState[ActorStateT],
        Dict[str, Any] # All observations
    ]:
        state, common_state, actor_state, onboard_obs = self.onboard_reset(
            state=state,
            common_state=common_state,
            actor_state=combined_state.actor_state
        )
        
        sensor_obss = {}
        sensor_states = {}
        for key, sensor in self._sensors.items():
            state, common_state, sensor_state, sensor_obs = sensor.reset(
                state=state,
                common_state=common_state,
                sensor_state=combined_state.sensor_states[key]
            )
            sensor_obss[key] = sensor_obs
            sensor_states[key] = sensor_state
        
        combined_obs = onboard_obs.copy()
        combined_obs.update(sensor_obss)
        return (
            state,
            common_state,
            FuncActorCombinedState(
                actor_state=actor_state, 
                sensor_states=sensor_states
            ),
            combined_obs
        )

    def step(
        self,
        state : StateType,
        common_state : FuncEnvCommonState[BDeviceType, BRNGType],
        combined_state : FuncActorCombinedState[ActorStateT],
        action : ActorActT
    ) -> Tuple[
        StateType,
        FuncEnvCommonState[BDeviceType, BRNGType],
        FuncActorCombinedState[ActorStateT],
        Dict[str, Any] # All observations
    ]:
        state, common_state, actor_state, onboard_obs = self.onboard_step(
            state=state,
            common_state=common_state,
            actor_state=combined_state.actor_state,
            action=action
        )
        
        sensor_obss = {}
        sensor_states = {}
        for key, sensor in self._sensors.items():
            state, common_state, sensor_state, sensor_obs = sensor.step(
                state=state,
                common_state=common_state,
                sensor_state=combined_state.sensor_states[key]
            )
            sensor_obss[key] = sensor_obs
            sensor_states[key] = sensor_state
        
        combined_obs = onboard_obs.copy()
        combined_obs.update(sensor_obss)
        return (
            state,
            common_state,
            FuncActorCombinedState(
                actor_state=actor_state, 
                sensor_states=sensor_states
            ),
            combined_obs
        )
    
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
        
        for key, sensor in self._sensors.items():
            state, common_state = sensor.close(
                state=state,
                common_state=common_state,
                sensor_state=combined_state.sensor_states[key]
            )
        
        return state, common_state