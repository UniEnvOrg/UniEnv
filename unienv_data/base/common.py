from typing import List, Tuple, Union, Dict, Any, Optional, Generic, TypeVar
import os
import abc
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.env_base.env import ContextType, ObsType, ActType
from unienv_interface.space import Space, batch_utils as space_batch_utils
import dataclasses

class TransitionsBase(abc.ABC, Generic[BArrayType, BDeviceType, BDtypeType, BRNGType, ContextType, ObsType, ActType]):
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType] = None

    obs_space : Space[ObsType, Any, BDeviceType, BDtypeType, BRNGType]
    act_space : Space[ActType, Any, BDeviceType, BDtypeType, BRNGType]
    context_space : Optional[Space[ContextType, Any, BDeviceType, BDtypeType, BRNGType]]
    metadata_space : Optional[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]] = None

    def __len__(self) -> int:
        return self.len_transitions()

    @abc.abstractmethod
    def len_transitions(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_context(self, idx : Union[int, slice, BArrayType]) -> Optional[BArrayType]:
        """
        Get a slice of the transitions.
        idx can be an integer index, a slice object, a boolean mask tensor, or a 1d integer tensor (list of index)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_obs(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        """
        Get a slice of the transitions.
        idx can be an integer index, a slice object, a boolean mask tensor, or a 1d integer tensor (list of index)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_next_obs(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        """
        Get a slice of the transitions.
        idx can be an integer index, a slice object, a boolean mask tensor, or a 1d integer tensor (list of index)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_act(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        """
        Get a slice of the transitions.
        idx can be an integer index, a slice object, a boolean mask tensor, or a 1d integer tensor (list of index)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_reward(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        """
        Get a slice of the transitions.
        idx can be an integer index, a slice object, a boolean mask tensor, or a 1d integer tensor (list of index)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_termination(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        """
        Get a slice of the transitions.
        idx can be an integer index, a slice object, a boolean mask tensor, or a 1d integer tensor (list of index)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_truncation(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        """
        Get a slice of the transitions.
        idx can be an integer index, a slice object, a boolean mask tensor, or a 1d integer tensor (list of index)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_episode_ids(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        """
        Get a slice of the transitions.
        idx can be an integer index, a slice object, a boolean mask tensor, or a 1d integer tensor (list of index)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_metadata(self, idx : Union[int, slice, BArrayType]) -> Optional[BArrayType]:
        """
        Get a slice of the transitions.
        idx can be an integer index, a slice object, a boolean mask tensor, or a 1d integer tensor (list of index)
        """
        raise NotImplementedError
    

class Transitions(TransitionsBase[BArrayType, BDeviceType, BDtypeType, BRNGType, ContextType, ObsType, ActType]):
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType] = None

    """
    The observation space, action space, and context space are just stored here for reference. 
    We will always store the flattened arrays because they are easier to work with.
    If user need to access the actual observation, action, and context values, they should call `unflatten` or `batch_unflatten` on the corresponding values.
    """
    obs_space : Space[ObsType, Any, BDeviceType, BDtypeType, BRNGType]
    act_space : Space[ActType, Any, BDeviceType, BDtypeType, BRNGType]
    context_space : Optional[Space[ContextType, Any, BDeviceType, BDtypeType, BRNGType]]
    metadata_space : Optional[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]] = None

    context : Optional[BArrayType] # Shape: (len_transitions, flattened_context_shape)
    obs : BArrayType # Shape: (len_transitions, flattened_obs_shape)
    act : BArrayType # Shape: (len_transitions, flattened_act_shape)
    next_obs : BArrayType # Shape: (len_transitions, flattened_obs_shape)
    reward : BArrayType # Shape: (len_transitions, )
    termination : BArrayType # Shape: (len_transitions, )
    truncation : BArrayType # Shape: (len_transitions, )
    episode_ids : BArrayType # Shape: (len_transitions, )
    metadata : Optional[BArrayType] # Shape: (len_transitions, flattened_metadata_shape)

    sample_ids : Optional[BArrayType] = None # Shape: (len_transitions, ), this is only used by a sampler to keep track of the sample ids

    def __init__(
        self,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device : Optional[BDeviceType],
        
        obs_space : Space[ObsType, Any, BDeviceType, BDtypeType, BRNGType],
        act_space : Space[ActType, Any, BDeviceType, BDtypeType, BRNGType],
        context_space : Optional[Space[ContextType, Any, BDeviceType, BDtypeType, BRNGType]],
        metadata_space : Optional[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]],
        
        context : Optional[BArrayType],
        obs : BArrayType,
        act : BArrayType,
        next_obs : BArrayType,
        reward : BArrayType,
        termination : BArrayType,
        truncation : BArrayType,
        episode_ids : BArrayType,
        metadata : Optional[BArrayType],
        
        sample_ids : Optional[BArrayType] = None
    ):
        assert (
            obs_space.backend == backend and 
            act_space.backend == backend and 
            (context_space is None or context_space.backend == backend) and 
            (metadata_space is None or metadata_space.backend == backend)
        )
        assert (metadata is None) == (metadata_space is None)
        assert (context is None) == (context_space is None)
        
        self.backend = backend
        self.device = device
        if device is not None:
            obs_space = obs_space.to_device(device)
            act_space = act_space.to_device(device)
            if context_space is not None:
                context_space = context_space.to_device(device)
            if metadata_space is not None:
                metadata_space = metadata_space.to_device(device)
        self.obs_space = obs_space
        self.act_space = act_space
        self.context_space = context_space
        self.metadata_space = metadata_space
        
        def tensor_device_transform(x):
            if device is not None and x is not None:
                return backend.to_device(x, device)
            return x
        self.context = tensor_device_transform(context)
        self.obs = tensor_device_transform(obs)
        self.act = tensor_device_transform(act)
        self.next_obs = tensor_device_transform(next_obs)
        self.reward = tensor_device_transform(reward)
        self.termination = tensor_device_transform(termination)
        self.truncation = tensor_device_transform(truncation)
        self.episode_ids = tensor_device_transform(episode_ids)
        self.metadata = tensor_device_transform(metadata) if metadata is not None else None
        
        self.sample_ids = tensor_device_transform(sample_ids) if sample_ids is not None else None

    def __len__(self) -> int:
        return self.len_transitions()

    def len_transitions(self) -> int:
        return self.reward.shape[0]
    
    
    def get_context(self, idx : Union[int, slice, BArrayType]) -> Optional[BArrayType]:
        return self.context[idx] if self.context is not None else None
    
    def get_obs(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        return self.obs[idx]
    
    def get_next_obs(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        return self.next_obs[idx]
    
    def get_act(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        return self.act[idx]
    
    def get_reward(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        return self.reward[idx]
    
    def get_termination(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        return self.termination[idx]
    
    def get_truncation(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        return self.truncation[idx]
    
    def get_episode_ids(self, idx : Union[int, slice, BArrayType]) -> BArrayType:
        return self.episode_ids[idx]
    
    def get_metadata(self, idx : Union[int, slice, BArrayType]) -> Optional[BArrayType]:
        return self.metadata[idx] if self.metadata is not None else None


class TrajectorySlice(Transitions[BArrayType, BDeviceType, BDtypeType, BRNGType, ContextType, ObsType, ActType]):
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
    device : Optional[BDeviceType] = None

    obs_space : Space[ObsType, Any, BDeviceType, BDtypeType, BRNGType]
    act_space : Space[ActType, Any, BDeviceType, BDtypeType, BRNGType]
    context_space : Optional[Space[ContextType, Any, BDeviceType, BDtypeType, BRNGType]]
    metadata_space : Optional[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]] = None

    start_context : Optional[BArrayType] # Shape: (flattened_context_shape, )
    obs : BArrayType # Shape: (len_transitions, flattened_obs_shape)
    final_next_obs : BArrayType # Shape: (flattened_obs_shape, )
    act : BArrayType # Shape: (len_transitions, flattened_act_shape)
    reward : BArrayType # Shape: (len_transitions, )
    termination : BArrayType # Shape: (len_transitions, )
    truncation : BArrayType # Shape: (len_transitions, )
    metadata : Optional[BArrayType] # Shape: (len_transitions, flattened_metadata_shape)

    sample_ids : Optional[BArrayType] = None # Shape: (len_transitions, ), this is only used by a sampler to keep track of the sample ids

    def __init__(
        self,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        device : Optional[BDeviceType],

        obs_space : Space[ObsType, Any, BDeviceType, BDtypeType, BRNGType],
        act_space : Space[ActType, Any, BDeviceType, BDtypeType, BRNGType],
        context_space : Optional[Space[ContextType, Any, BDeviceType, BDtypeType, BRNGType]],
        metadata_space : Optional[Space[Any, Any, BDeviceType, BDtypeType, BRNGType]],

        start_context : Optional[BArrayType],
        obs : BArrayType,
        final_next_obs : BArrayType,
        act : BArrayType,
        reward : BArrayType,
        termination : BArrayType,
        truncation : BArrayType,
        metadata : Optional[BArrayType],

        sample_ids : Optional[BArrayType] = None
    ):
        assert (
            obs_space.backend == backend and 
            act_space.backend == backend and 
            (context_space is None or context_space.backend == backend) and 
            (metadata_space is None or metadata_space.backend == backend)
        )
        assert (metadata is None) == (metadata_space is None)
        assert (start_context is None) == (context_space is None)

        self.backend = backend
        self.device = device
        if device is not None:
            obs_space = obs_space.to_device(device)
            act_space = act_space.to_device(device)
            if context_space is not None:
                context_space = context_space.to_device(device)
            if metadata_space is not None:
                metadata_space = metadata_space.to_device(device)
        self.obs_space = obs_space
        self.act_space = act_space
        self.context_space = context_space
        self.metadata_space = metadata_space

        def tensor_device_transform(x):
            if device is not None and x is not None:
                return backend.to_device(x, device)
            return x
        self.start_context = tensor_device_transform(start_context)
        self.obs = tensor_device_transform(obs)
        self.final_next_obs = tensor_device_transform(final_next_obs)
        self.act = tensor_device_transform(act)
        self.reward = tensor_device_transform(reward)
        self.termination = tensor_device_transform(termination)
        self.truncation = tensor_device_transform(truncation)
        self.metadata = tensor_device_transform(metadata) if metadata is not None else None
        
        self.sample_ids = tensor_device_transform(sample_ids) if sample_ids is not None else None

    def __len__(self) -> int:
        return self.len_transitions()

    def len_transitions(self) -> int:
        return self.reward.shape[0]
    
    @property
    def context(self) -> Optional[ContextType]:
        if self.start_context is None:
            return None
        return self.backend.array_api_namespace.broadcast_to(self.start_context, (self.obs.shape[0],) + self.start_context.shape)

    @property
    def next_obs(self) -> ObsType:
        return self.backend.array_api_namespace.concat([
            self.obs[1:], self.final_next_obs[None]
        ], axis=0)

    @property
    def episode_ids(self) -> BArrayType:
        return self.backend.array_api_namespace.zeros(self.reward.shape, dtype=self.backend.default_integer_dtype)

    
SamplerArrayType = TypeVar('SamplerArrayType')
SamplerDeviceType = TypeVar('SamplerDeviceType')
SamplerDtypeType = TypeVar('SamplerDtypeType')
SamplerRNGType = TypeVar('SamplerRNGType')
SamplerContextType = TypeVar('SamplerContextType')
SamplerObsType = TypeVar('SamplerObsType')
SamplerActType = TypeVar('SamplerActType')
class TransitionSampler(abc.ABC, Generic[
    SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, SamplerContextType, SamplerObsType, SamplerActType,
    BArrayType, BDeviceType, BDtypeType, BRNGType, ContextType, ObsType, ActType
]):
    batch_size : int

    backend : ComputeBackend[SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]
    device : Optional[SamplerDeviceType] = None

    obs_space : Space[SamplerObsType, Any, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]
    act_space : Space[SamplerActType, Any, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]
    context_space : Optional[Space[SamplerContextType, Any, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]]
    metadata_space : Optional[Space[Any, Any, SamplerDeviceType, SamplerDtypeType, SamplerRNGType]] = None

    data : TransitionsBase[BArrayType, BDeviceType, BDtypeType, BRNGType, ContextType, ObsType, ActType]

    @abc.abstractmethod
    def sample(self) -> Transitions[
        SamplerArrayType, SamplerDeviceType, SamplerDtypeType, SamplerRNGType, SamplerContextType, SamplerObsType, SamplerActType
    ]:
        """
        Should sample a batch of transitions from the data.
        The transitions will have the `sample_ids` filled with values from 0 to `batch_size`.
        """
        raise NotImplementedError
    
