from typing import Any, Generic, TypeVar, Optional, Dict, Tuple, Sequence, List, Type, Union, Callable, SupportsFloat
from dataclasses import dataclass, replace as dataclass_replace
from unienv_interface.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.space import Space, Dict as DictSpace
from unienv_interface.env_base.env import Env
from unienv_interface.env_base.funcenv import FuncEnv, FuncEnvCommonRenderState
from .world import World, FuncWorld
from .sensor import Sensor, FuncSensor
from .actor import Actor, FuncActor, ActorStateT, FuncActorCombinedState
from .sensors import CameraSensor, FuncCameraSensor, WindowedViewSensor, FuncWindowedViewSensor
from .task import Task, FuncTask, TaskStateT

WorldStateT = TypeVar("WorldStateT")
RenderSensorStateT = TypeVar("RenderSensorStateT")
RenderSensorDataT = TypeVar("RenderSensorDataT")

@dataclass(frozen=True)
class WorldBasedFuncEnvState(
    Generic[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT]
):
    world_state : WorldStateT
    actor_state : FuncActorCombinedState[ActorStateT]
    task_state : Optional[TaskStateT] = None
    render_sensor_state : Optional[RenderSensorStateT] = None
    render_sensor_data : Optional[RenderSensorDataT] = None
    render_elapsed : float = 0.0

@dataclass(frozen=True)
class WorldBasedFuncEnvRenderState(
    Generic[RenderSensorDataT]
):
    render_sensor_data : Optional[RenderSensorDataT]

WorldBasedFuncEnvInfoCallback = Callable[[
    FuncWorld[WorldStateT, BDeviceType, BDtypeType, BRNGType],
    FuncActor[WorldStateT, ActorStateT, BDeviceType, BDtypeType, BRNGType],
    WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
    BRNGType,
    Dict[str, Any], # Observation
    Optional[Dict[str, Any]], # Optional Context
    Optional[Any], # Optional Action (When stepping)
], Union[
    Dict[str, Any],
    Sequence[Dict[str, Any]]
]]

class WorldBasedFuncEnv(
    FuncEnv[
        WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
        WorldBasedFuncEnvRenderState[RenderSensorDataT], 
        BArrayType,
        Optional[Dict[str, Any]], # Context
        Dict[str, Any], # Observation
        Any, # Action
        RenderSensorDataT,
        BDeviceType,
        BDtypeType,
        BRNGType
    ],
    Generic[
        WorldStateT,
        ActorStateT,
        TaskStateT,
        RenderSensorStateT,
        RenderSensorDataT,
        BArrayType,
        BDeviceType,
        BDtypeType,
        BRNGType
    ]
):
    def __init__(
        self,
        world : FuncWorld[WorldStateT, BDeviceType, BDtypeType, BRNGType],
        actor : FuncActor[WorldStateT, ActorStateT, BDeviceType, BDtypeType, BRNGType],
        task : FuncTask[WorldStateT, ActorStateT, TaskStateT, BArrayType, BDeviceType, BDtypeType, BRNGType],
        render_sensor : Optional[Union[
            FuncSensor[WorldStateT, RenderSensorStateT, RenderSensorDataT, BDeviceType, BDtypeType, BRNGType],
            str
        ]] = None, # Either a sensor or the name of the attached sensor
        info_callback : Optional[WorldBasedFuncEnvInfoCallback] = None
    ):
        if render_sensor in actor.sensors.values():
            raise ValueError("The render sensor should not be attached to the actor. If you want to use the actor's sensors, use the actor's sensor names.")
        assert actor.backend == world.backend, "The actor and the world should have the same backend."

        self.batch_size = None # TODO: Implement batched environment
        self.world = world
        self.actor = actor
        self.task = task
        self.render_sensor = render_sensor
        self.info_callback = info_callback

        assert world.world_timestep is None or actor.control_timestep % world.world_timestep == 0, "The actor's control timestep should be dividable by the world's timestep."
        self._n_world_step = int(actor.control_timestep / world.world_timestep) if world.world_timestep is not None else None
        self._need_update_render_sensor = render_sensor is not None and not isinstance(render_sensor, str)
        real_render_sensor = self.get_render_sensor_instance()
        if real_render_sensor is None:
            self.render_mode = None
        elif isinstance(real_render_sensor, CameraSensor):
            self.render_mode = real_render_sensor.camera_mode
        else:
            self.render_mode = "human"
        
        self.metadata = {
            "render_modes": [] if self.render_mode is None else [self.render_mode]
        }

    @property
    def backend(self) -> ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]:
        return self.world.backend

    @property
    def device(self) -> BDeviceType:
        return self.actor.device

    @property
    def observation_space(self) -> DictSpace[BDeviceType, BDtypeType, BRNGType]:
        if self.task.observation_space is None:
            return self.actor.observation_space

        new_spaces = self.actor.observation_space.spaces.copy()
        new_spaces.update(self.task.observation_space.spaces)
        return DictSpace(
            backend=self.backend,
            spaces=new_spaces,
            device=self.actor.device,
        )
    
    @property
    def context_space(self) -> Optional[DictSpace[BDeviceType, BDtypeType, BRNGType]]:
        return self.task.context_space
    
    @property
    def action_space(self) -> Space[Any, Any, BDeviceType, BDtypeType, BRNGType]:
        return self.actor.action_space

    def get_render_sensor_instance(self) -> Optional[FuncSensor[WorldStateT, RenderSensorStateT, RenderSensorDataT, BDeviceType, BDtypeType, BRNGType]]:
        if self.render_sensor is None:
            return None
        elif isinstance(self.render_sensor, str):
            return self.actor.sensors[self.render_sensor]
        else:
            return self.render_sensor
    
    def initial(
        self,
        rng : BRNGType,
        world_kwargs : Dict[str, Any] = {},
        actor_kwargs : Dict[str, Any] = {},
        task_kwargs : Dict[str, Any] = {},
        render_sensor_kwargs : Dict[str, Any] = {},
        device : Optional[BDeviceType] = None
    ) -> Tuple[
        WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
        BRNGType,
        Optional[Dict[str, Any]], # Context
        Dict[str, Any], # Observation
        Dict[str, Any] # Info
    ]:
        world_state, rng = self.world.initial(
            rng, **world_kwargs
        )
        world_state, rng, actor_combined_state = self.actor.initial(
            self.world, world_state, rng, **actor_kwargs
        )
        world_state, rng, actor_combined_state, task_state, task_context = self.task.initial(
            self.world, self.actor, world_state, rng, actor_combined_state, **task_kwargs
        )
        if self._need_update_render_sensor:
            world_state, rng, render_sensor_state = self.get_render_sensor_instance().initial(
                self.world, world_state, rng, **render_sensor_kwargs
            )
        else:
            render_sensor_state = None
        world_state, rng, actor_combined_state, obs = self.actor.get_data(
            world_state,
            rng,
            actor_combined_state,
            0.0
        )
        if isinstance(self.render_sensor, str):
            render_sensor_data = obs[self.render_sensor]
        else:
            render_sensor_data = None

        if self.task.observation_space is not None:
            task_obs = self.task.get_data(
                world_state, rng, actor_combined_state, obs, task_state, 0.0
            )
            obs.update(task_obs)
        
        env_state = WorldBasedFuncEnvState(
            world_state=world_state,
            actor_state=actor_combined_state,
            task_state=task_state,
            render_sensor_state=render_sensor_state,
            render_sensor_data=render_sensor_data,
            render_elapsed=0.0
        )
        if self.info_callback is not None:
            info = self.info_callback(
                self.world,
                self.actor,
                env_state,
                rng,
                obs,
                task_context,
                None # No action
            )
        else:
            info = {}
        return env_state, rng, task_context, obs, info

    def reset(
        self,
        state : WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
        rng : BRNGType,
        mask : Optional[BArrayType] = None
    ) -> Tuple[
        WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
        BRNGType,
        Optional[Dict[str, Any]], # Context
        Dict[str, Any], # Observation
        Dict[str, Any] # Info
    ]:
        assert mask is None, "The batched environment is not supported yet."
        world_state, actor_combined_state, task_state, render_sensor_state = state.world_state, state.actor_state, state.task_state, state.render_sensor_state
        world_state, rng = self.world.reset(
            world_state, rng
        )
        world_state, rng, actor_combined_state = self.actor.reset(
            self.world, world_state, rng, actor_combined_state
        )
        world_state, rng, actor_combined_state, task_state, task_context = self.task.reset(
            self.world, self.actor, world_state, rng, actor_combined_state, task_state
        )
        if self._need_update_render_sensor:
            world_state, rng, render_sensor_state = self.get_render_sensor_instance().reset(
                self.world, world_state, rng, render_sensor_state
            )
        else:
            render_sensor_state = None
        
        world_state, rng, actor_combined_state, obs = self.actor.get_data(
            world_state,
            rng,
            actor_combined_state,
            0.0
        )

        if isinstance(self.render_sensor, str):
            render_sensor_data = obs[self.render_sensor]
        else:
            render_sensor_data = None
        
        if self.task.observation_space is not None:
            task_obs = self.task.get_data(
                world_state, rng, actor_combined_state, obs, task_state, 0.0
            )
            obs.update(task_obs)

        env_state = WorldBasedFuncEnvState(
            world_state=world_state,
            actor_state=actor_combined_state,
            task_state=task_state,
            render_sensor_state=render_sensor_state,
            render_sensor_data=render_sensor_data,
            render_elapsed=0.0
        )
        if self.info_callback is not None:
            info = self.info_callback(
                self.world,
                self.actor,
                env_state,
                rng,
                obs,
                task_context,
                None # No action
            )
        else:
            info = {}
        return env_state, rng, task_context, obs, info

    def step(
        self,
        state : WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
        rng : BRNGType,
        action : Any
    ) -> Tuple[
        WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
        BRNGType,
        Dict[str, Any],
        Union[SupportsFloat, BArrayType],
        Union[bool, BArrayType],
        Union[bool, BArrayType],
        Dict[str, Any]
    ]:
        world_state, actor_combined_state, task_state, render_sensor_state = state.world_state, state.actor_state, state.task_state, state.render_sensor_state
        if self._n_world_step is None:
            total_elapsed = 0.0
            while total_elapsed < self.actor.control_timestep:
                elapsed_seconds, world_state, rng = self.world.step(
                    world_state, rng
                )
                total_elapsed += elapsed_seconds
                world_state, rng, actor_combined_state = self.actor.step(
                    world_state, rng, actor_combined_state, elapsed_seconds
                )
                world_state, rng, actor_combined_state, task_state = self.task.step(
                    world_state, rng, actor_combined_state, task_state, elapsed_seconds
                )
                if self._need_update_render_sensor:
                    world_state, rng, render_sensor_state = self.get_render_sensor_instance().step(
                        world_state, rng, render_sensor_state, elapsed_seconds
                    )
        else:
            total_elapsed = self.actor.control_timestep
            for i in range(self._n_world_step):
                elapsed_seconds, world_state, rng = self.world.step(
                    world_state, rng
                )
                world_state, rng, actor_combined_state = self.actor.step(
                    world_state, rng, actor_combined_state, elapsed_seconds
                )
                world_state, rng, actor_combined_state, task_state = self.task.step(
                    world_state, rng, actor_combined_state, task_state, elapsed_seconds
                )
                if self._need_update_render_sensor:
                    world_state, rng, render_sensor_state = self.get_render_sensor_instance().step(
                        world_state, rng, render_sensor_state, elapsed_seconds
                    )
        
        world_state, rng, actor_state = self.actor.set_next_action(
            world_state,
            rng,
            actor_combined_state.actor_state,
            action,
            total_elapsed
        )
        actor_combined_state = dataclass_replace(
            actor_combined_state,
            actor_state=actor_state
        )
        world_state, rng, actor_combined_state, obs = self.actor.get_data(
            world_state,
            rng,
            actor_combined_state,
            total_elapsed
        )
        if isinstance(self.render_sensor, str):
            render_sensor_data = obs[self.render_sensor]
        else:
            render_sensor_data = None

        if self.task.observation_space is not None:
            task_obs = self.task.get_data(
                world_state, rng, actor_combined_state, obs, task_state, total_elapsed
            )
            obs.update(task_obs)
        
        world_state, rng, actor_combined_state, task_state, reward, task_termination, task_truncation = self.task.control_step(
            world_state, rng, actor_combined_state, obs, task_state, total_elapsed
        )
        
        env_state = WorldBasedFuncEnvState(
            world_state=world_state,
            actor_state=actor_combined_state,
            task_state=task_state,
            render_sensor_state=render_sensor_state,
            render_sensor_data=render_sensor_data,
            render_elapsed=state.render_elapsed + total_elapsed
        )
        if self.info_callback is not None:
            info = self.info_callback(
                self.world,
                self.actor,
                env_state,
                rng,
                obs,
                None, # No context
                action
            )
        else:
            info = {}
        
        return env_state, rng, obs, reward, task_termination, task_truncation, info

    def close(
        self,
        state : WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
        rng : BRNGType,
    ) -> None:
        world_state, actor_combined_state, task_state, render_sensor_state = state.world_state, state.actor_state, state.task_state, state.render_sensor_state
        world_state, rng, actor_combined_state = self.task.close(
            world_state, rng, actor_combined_state, task_state
        )
        if self._need_update_render_sensor:
            world_state, rng = self.get_render_sensor_instance().close(
                world_state, rng, render_sensor_state
            )
        world_state, rng = self.actor.close(
            world_state, rng, actor_combined_state
        )
        self.world.close(
            world_state, rng
        )
    
    def render_init(
        self, 
        state: WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT], 
        rng: BRNGType, 
        *, 
        render_mode: str | None = None
    ) -> Tuple[
        WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
        BRNGType,
        WorldBasedFuncEnvRenderState[RenderSensorDataT],
        FuncEnvCommonRenderState
    ]:
        assert render_mode == self.render_mode or render_mode is None, "The render mode should be the same as the render sensor's mode."
        assert self.render_sensor is not None, "The render sensor is not available."

        if self._need_update_render_sensor:
            world_state, render_sensor_state = state.world_state, state.render_sensor_state
            real_render_sensor = self.get_render_sensor_instance()
            world_state, rng, render_sensor_state, render_sensor_data = real_render_sensor.get_data(
                world_state, rng, render_sensor_state, state.render_elapsed
            )
            return dataclass_replace(
                state,
                world_state=world_state,
                render_sensor_state=render_sensor_state,
                render_sensor_data=None,
                render_elapsed=0.0
            ), rng, WorldBasedFuncEnvRenderState(
                render_sensor_data=render_sensor_data
            ), FuncEnvCommonRenderState(
                render_mode=render_mode,
                render_fps=int(round(1/self.actor.control_timestep))
            )
        else:
            return dataclass_replace(
                state,
                render_elapsed=0.0
            ), rng, WorldBasedFuncEnvRenderState(
                render_sensor_data=None
            ), FuncEnvCommonRenderState(
                render_mode=render_mode,
                render_fps=int(round(1/self.actor.control_timestep))
            )
    
    def render_image(
        self, 
        state: WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT], 
        rng: BRNGType, 
        render_state: WorldBasedFuncEnvRenderState[RenderSensorDataT], 
        render_rng: FuncEnvCommonRenderState
    ) -> Tuple[
        RenderSensorDataT | Sequence[RenderSensorDataT] | None,
        WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT], 
        BRNGType, 
        WorldBasedFuncEnvRenderState[RenderSensorDataT],
        FuncEnvCommonRenderState
    ]:
        if self._need_update_render_sensor:
            world_state, render_sensor_state = state.world_state, state.render_sensor_state
            real_render_sensor = self.get_render_sensor_instance()
            
            world_state, rng, render_sensor_state, render_sensor_data = real_render_sensor.get_data(
                world_state, rng, render_sensor_state, state.render_elapsed
            )
            return render_sensor_data, dataclass_replace(
                state,
                world_state=world_state,
                render_sensor_state=render_sensor_state,
                render_sensor_data=None,
                render_elapsed=0.0
            ), rng, dataclass_replace(
                render_state,
                render_sensor_data=render_sensor_data
            ), render_rng
        else:
            return state.render_sensor_data, dataclass_replace(
                state,
                render_elapsed=0.0,
            ), rng, render_state, render_rng

    def render_close(
        self, 
        state: WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT], 
        rng: BRNGType, 
        render_state: WorldBasedFuncEnvRenderState[RenderSensorDataT], 
        render_rng: FuncEnvCommonRenderState
    ) -> Tuple[
        WorldBasedFuncEnvState[WorldStateT, ActorStateT, TaskStateT, RenderSensorStateT],
        BRNGType
    ]:
        return state, rng