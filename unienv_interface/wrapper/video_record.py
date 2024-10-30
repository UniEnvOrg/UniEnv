from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal
from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.backends.base import ComputeBackend, BDtypeType, BRNGType, BDeviceType
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame
import os
import numpy as np

"""
This wrapper will accumulate the render frames of each step in the episode, and store it in the `episodic_frames` attribute.
"""
class EpisodeRenderStackWrapper(
    Wrapper[
        ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType,
        ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType
    ]
):
    def __init__(
        self,
        env : Env[ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType]
    ):
        super().__init__(env)
        self._step_frame = None
        self.episodic_frames = []
        self.video_episode_num = 0
        self.video_step_num = 0
        self.video_episode_start_step = 0

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType, 
        RewardType, 
        TerminationType, 
        TerminationType, 
        Dict[str, Any]
    ]:
        self._step_frame = None
        self.video_step_num += 1
        obs, rew, termination, truncation, info = self.env.step(action)
        self._post_step_render()
        if self.env.backend.array_api_namespace.any(self.env.backend.array_api_namespace.logical_or(
            termination, truncation
        )):
            self._post_episode()
        return obs, rew, termination, truncation, info
    
    def _post_step_render(self) -> None:
        self._step_frame = self.env.render()
        self.episodic_frames.append(self._step_frame)

    def _post_episode(self) -> None:
        pass

    def reset(
        self,
        *args,
        seed : Optional[int] = None,
        **kwargs
    ) -> Tuple[ContextType, ObsType, Dict[str, Any]]:
        if self.episodic_frames is not None:
            self._post_episode()
        if len(self.episodic_frames) > 1:
            self.video_episode_num += 1
        self.episodic_frames = []
        self.video_episode_start_step = self.video_step_num
        self._step_frame = None
        ret = self.env.reset(*args, seed=seed, **kwargs)
        self._post_step_render()
        return ret
    
    def render(
        self
    ) -> Optional[Union[RenderFrame, Sequence[RenderFrame]]]:
        if self._step_frame is None:
            self._step_frame = self.env.render()
        return self._step_frame

class EpisodeVideoWrapper(
    EpisodeRenderStackWrapper[
        ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType
    ]
):
    def __init__(
        self, 
        env: Env[ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType],
        store_dir: os.PathLike | str,
        store_format : Literal['mp4', 'gif', 'webm'] = 'webm'
    ):
        super().__init__(env)
        store_dir = os.path.abspath(store_dir)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        self.store_dir = store_dir
        self.store_format = store_format

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
            frame_np = self.env.backend.to_numpy(frame)
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
        ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType
    ]
):
    def __init__(
        self, 
        env: Env[ContextType, ObsType, ActType, RewardType, TerminationType, RenderFrame, BDeviceType, BRNGType],
        wandb_log_key: str,
        store_format : Literal['mp4', 'gif', 'webm'] = 'webm',
        control_wandb_step: bool = False # Whether to auto-increment the wandb log step
    ):
        super().__init__(env)
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                'wandb is not installed, run `pip install wandb`'
            ) from e
        
        self.wandb = wandb
        self.wandb_log_key = wandb_log_key
        self.control_wandb_step = control_wandb_step
        self.store_format = store_format

    def _post_episode(self) -> None:
        if len(self.episodic_frames) <= 1:
            return
        
        assert self.episodic_frames[0].shape[2] == 3
        frames = np.zeros((
            len(self.episodic_frames),
            *self.episodic_frames[0].shape
        ))
        for i, frame in enumerate(self.episodic_frames):
            frame_np = self.env.backend.to_numpy(frame)
            frames[i] = frame_np
        clip = self.wandb.Video(
            frames.transpose(0, 3, 1, 2),
            fps=self.env.render_fps or 30,
            format=self.store_format
        )
        if self.control_wandb_step:
            self.wandb.log({self.wandb_log_key: clip, self.wandb_log_key + 'episode_id': self.video_episode_num}, step=self.video_episode_start_step)
        else:
            self.wandb.log({self.wandb_log_key: clip, self.wandb_log_key + '_episode_id': self.video_episode_num}, commit=False)


