from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal, SupportsFloat
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
import os
import numpy as np

"""
This wrapper will accumulate the render frames of each step in the episode, and store it in the `episodic_frames` attribute.
"""
class EpisodeRenderStackWrapper(
    Wrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType,
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        *,
        rendered_env_index : Optional[int] = None
    ):
        """
        Initialize the wrapper.
        For a batched environment, you can specify the `rendered_env_index`, which is used to check if the episode for the rendering environment has ended.
        """
        super().__init__(env)
        self.rendered_env_index = rendered_env_index

        self._step_frame = None # Cache the frame of the current step
        self.episodic_frames = []
        self._has_post_episode = False

        # step info accumulator
        self.video_episode_num = 0
        self.video_step_num = 0
        self.video_episode_start_step = 0

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType, 
        Union[SupportsFloat, BArrayType], 
        Union[bool, BArrayType], 
        Union[bool, BArrayType],
        Dict[str, Any]
    ]:
        self._step_frame = None
        self.video_step_num += 1
        obs, rew, termination, truncation, info = self.env.step(action)
        self._post_step_render()
        
        if not self._has_post_episode and self._is_render_termination(termination, truncation):
            self._post_episode()
            self._has_post_episode = True
        
        return obs, rew, termination, truncation, info
    
    def _is_render_termination(
        self,
        termination: Union[bool, BArrayType],
        truncation: Union[bool, BArrayType]
    ) -> bool:
        if self.env.batch_size is None:
            return termination or truncation
        else:
            if self.rendered_env_index is None:
                return self.env.backend.any(
                    self.env.backend.logical_or(
                        termination, truncation
                    )
                )
            else:
                return self.env.backend.any(
                    self.env.backend.logical_or(
                        termination[self.rendered_env_index], truncation[self.rendered_env_index]
                    )
                )

    def _post_step_render(self) -> None:
        self._step_frame = self.env.render()
        self.episodic_frames.append(self._step_frame)

    def _post_episode(self) -> None:
        pass

    def reset(
        self,
        *args,
        mask : Optional[BArrayType] = None,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        render_terminated = False
        if self.env.batch_size is None:
            render_terminated = True
        else:
            if mask is None:
                render_terminated = True
            else:
                if self.rendered_env_index is None:
                    render_terminated = self.env.backend.any(mask)
                else:
                    render_terminated = bool(mask[self.rendered_env_index])

        if render_terminated:
            if len(self.episodic_frames) > 1:
                if not self._has_post_episode:
                    self._post_episode()
                self.video_episode_num += 1
            
            self.episodic_frames = []
            self._has_post_episode = False
            self.video_episode_start_step = self.video_step_num
        
        ret = self.env.reset(*args, seed=seed, **kwargs)
        self._post_step_render()
        return ret
    
    def render(
        self
    ) -> Optional[Union[RenderFrame, Sequence[RenderFrame]]]:
        if self._step_frame is None:
            self._step_frame = self.env.render()
        return self._step_frame
    
    def close(self):
        self._post_episode()
        super().close()

class EpisodeVideoWrapper(
    EpisodeRenderStackWrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self, 
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        *,
        store_dir: os.PathLike | str,
        format : Literal['mp4', 'gif', 'webm'] = 'webm',
        rendered_env_index : Optional[int] = None
    ):
        super().__init__(env, rendered_env_index=rendered_env_index)
        store_dir = os.path.abspath(store_dir)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        self.store_dir = store_dir
        self.store_format = format

    def _post_episode(self) -> None:
        if len(self.episodic_frames) <= 1:
            return
        
        video_path = os.path.join(self.store_dir, f"episode_{self.video_episode_num}_step_{self.video_episode_start_step}.{self.store_format}")

        try:
           from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        except ImportError as e:
            raise ImportError(
                'MoviePy is not installed, run `pip install moviepy`'
            ) from e

        frames = []
        for frame in self.episodic_frames:
            if self.env.backend.is_backendarray(frame):
                frame_np = self.env.backend.to_numpy(frame)
            else:
                assert isinstance(frame, np.ndarray)
                frame_np = frame
            assert frame_np.shape[2] == 3
            frames.append(frame_np)
        clip = ImageSequenceClip(frames, fps=self.env.render_fps or 30)
        clip.write_videofile(
            video_path,
            logger=None,
        )
        clip.close()
    
class EpisodeWandbVideoWrapper(
    EpisodeRenderStackWrapper[
        BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType
    ]
):
    def __init__(
        self, 
        env: Env[BArrayType, ContextType, ObsType, ActType, RenderFrame, BDeviceType, BDtypeType, BRNGType],
        *,
        wandb_log_key: str,
        format : Literal['mp4', 'gif', 'webm'] = 'webm',
        control_wandb_step: bool = False, # Whether to auto-increment the wandb log step
        log_wandb_episode_id: bool = True, # Whether to log the episode id
        rendered_env_index : Optional[int] = None
    ):
        super().__init__(env, rendered_env_index=rendered_env_index)
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                'wandb is not installed, run `pip install wandb`'
            ) from e
        
        self.wandb = wandb
        self.wandb_log_key = wandb_log_key
        self.log_wandb_episode_id = log_wandb_episode_id
        self.control_wandb_step = control_wandb_step
        self.store_format = format

    def _post_episode(self) -> None:
        if len(self.episodic_frames) <= 1:
            return
        
        assert self.episodic_frames[0].shape[2] == 3, "Only RGB frames are supported for wandb video logging, got shape {}".format(self.episodic_frames[0].shape)
        frames = np.zeros((
            len(self.episodic_frames),
            *self.episodic_frames[0].shape
        ))
        for i, frame in enumerate(self.episodic_frames):
            if self.env.backend.is_backendarray(frame):
                frame_np = self.env.backend.to_numpy(frame)
            else:
                assert isinstance(frame, np.ndarray)
                frame_np = frame
            assert frame_np.shape[2] == 3
            frames[i] = frame_np
        clip = self.wandb.Video(
            frames.transpose(0, 3, 1, 2),
            fps=self.env.render_fps or 30,
            format=self.store_format
        )

        to_log = {
            self.wandb_log_key: clip
        }
        if self.log_wandb_episode_id:
            to_log[self.wandb_log_key + '_episode_id'] = self.video_episode_num
        
        if self.control_wandb_step:
            self.wandb.log(to_log, step=self.video_episode_start_step)
        else:
            self.wandb.log(to_log, commit=False)


