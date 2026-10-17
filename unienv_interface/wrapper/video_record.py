from typing import Dict, Any, Tuple, Optional, Sequence, Union, Generic, Literal, SupportsFloat, List
from collections.abc import Mapping
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

from unienv_interface.env_base.wrapper import Wrapper
from unienv_interface.env_base.env import Env, ContextType, ObsType, ActType, RenderFrame
from unienv_interface.space import Space
from unienv_interface.space.space_utils.metadata_utils import find_spaces_by_metadata
from unienv_interface.transformations.flatten_dict_transform import flatten_nested_mapping
import os
import numpy as np

# Flattened frame key (``None`` for a non-mapping frame) -> (layout, format) declared by the space metadata
ImageSpecs = Dict[Optional[str], Tuple[Optional[str], Optional[str]]]

def image_specs_from_space(
    space : Optional[Space],
    nested_separator : str = ".",
) -> Optional[ImageSpecs]:
    """The image leaves of ``space`` declared through ``metadata["kind"] == "image"``.

    Returns ``{flattened key: (layout, format)}`` (key ``None`` when ``space`` itself
    is the image), or ``None`` when no leaf of ``space`` declares a ``kind`` at all,
    i.e. the space carries no usable description and callers should fall back to
    treating every frame as an RGB image.
    """
    if space is None:
        return None
    leaves = find_spaces_by_metadata(space)
    if not any("kind" in leaf.metadata for _, leaf in leaves):
        return None
    specs : ImageSpecs = {}
    for path, leaf in leaves:
        if leaf.metadata.get("kind") == "image":
            key = nested_separator.join(str(p) for p in path) if len(path) > 0 else None
            specs[key] = (leaf.metadata.get("layout"), leaf.metadata.get("format"))
    return specs

def image_to_hwc_rgb(
    image : np.ndarray,
    layout : Optional[str] = None,
    format : Optional[str] = None,
) -> np.ndarray:
    """Convert an image described by ``layout`` / ``format`` metadata to ``(H, W, 3)`` RGB.

    ``layout`` ``"CHW"`` is moved channel-last; ``format`` ``"bgr"`` / ``"bgra"`` are
    reordered, ``"rgba"`` drops alpha and ``"gray"`` is repeated over three channels.
    ``None`` (or ``"HWC"`` / ``"rgb"``) leaves the image unchanged.
    """
    if layout is not None and layout.upper() == "CHW" and image.ndim >= 3:
        image = np.moveaxis(image, -3, -1)
    fmt = None if format is None else format.lower()
    if fmt == "bgr":
        image = image[..., ::-1]
    elif fmt == "bgra":
        image = image[..., 2::-1]
    elif fmt == "rgba":
        image = image[..., :3]
    elif fmt == "gray":
        if image.ndim == 2:
            image = image[..., None]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
    return np.ascontiguousarray(image)

def _looks_like_rgb_image(value : Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) == 3 and shape[-1] == 3

"""
This wrapper accumulates render frames for each step in the episode in `episodic_frames`.
If render returns a mapping, keys are flattened and the history is stored as a dict.
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
        done_index : Optional[int] = None,
        rendered_index : Optional[int] = None,
        nested_separator : str = ".",
        frame_source : Literal["render", "observation"] = "render",
        frame_space : Optional[Space] = None,
    ):
        """
        Initialize the wrapper.
        For a batched environment, `done_index` is used to check episode termination on a specific batch index.
        If `rendered_index` is provided, render outputs that are batched tensors are indexed at this batch index before caching/logging.

        Frames are taken from `env.render()` (`frame_source="render"`) or from the observations returned by
        `reset` / `step` (`frame_source="observation"`). `frame_space` describes the frames; it defaults to
        `env.observation_space` for observation frames. When its leaves declare `Space.metadata`, only the
        leaves with `kind == "image"` are recorded, converted to (H, W, 3) RGB according to their `layout`
        (`"CHW"` is moved channel-last) and `format` (`"bgr"`, `"rgba"`, `"gray"`, ...). Without any `kind`
        metadata, every render frame is recorded as an RGB image (and, for observation frames, every
        (H, W, 3) leaf).
        """
        super().__init__(env)
        assert frame_source in ("render", "observation"), f"Unsupported frame_source {frame_source!r}"
        self.done_index = done_index
        self.rendered_index = rendered_index
        self.nested_separator = nested_separator
        self.frame_source = frame_source
        if frame_space is None and frame_source == "observation":
            frame_space = env.observation_space
        self.frame_space = frame_space
        self._image_specs : Optional[ImageSpecs] = image_specs_from_space(frame_space, nested_separator)

        self._step_frame = None # Cache the frame of the current step
        self.episodic_frames : Union[List[Any], Dict[str, List[Any]]] = []
        self._episode_stack_mode : Optional[Literal["single", "mapping"]] = None
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
        self._post_step_render(obs)
        
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
            if self.done_index is None:
                return self.env.backend.any(
                    self.env.backend.logical_or(
                        termination, truncation
                    )
                )
            else:
                return self.env.backend.any(
                    self.env.backend.logical_or(
                        termination[self.done_index], truncation[self.done_index]
                    )
                )

    def _index_render_tensor_if_needed(self, frame: Any) -> Any:
        if self.rendered_index is None:
            return frame
        if self.env.backend.is_backendarray(frame) or isinstance(frame, np.ndarray):
            try:
                return frame[self.rendered_index]
            except Exception as e:
                raise ValueError(
                    f"Failed to index render tensor with rendered_index={self.rendered_index}. "
                    f"Frame type: {type(frame)}"
                ) from e
        return frame

    def _reset_episode_stack(self) -> None:
        self.episodic_frames = []
        self._episode_stack_mode = None

    def _episode_frame_count(self) -> int:
        if self._episode_stack_mode == "mapping":
            assert isinstance(self.episodic_frames, dict)
            if len(self.episodic_frames) == 0:
                return 0
            first_key = next(iter(self.episodic_frames.keys()))
            return len(self.episodic_frames[first_key])
        assert isinstance(self.episodic_frames, list)
        return len(self.episodic_frames)

    def _append_single_frame(self, frame: Any) -> None:
        if self._episode_stack_mode is None:
            self._episode_stack_mode = "single"
            self.episodic_frames = []
        elif self._episode_stack_mode != "single":
            raise ValueError(
                "Render frame type changed within an episode: expected mapping render frames, got non-mapping render frame."
            )
        assert isinstance(self.episodic_frames, list)
        self.episodic_frames.append(frame)

    def _append_flattened_mapping_frame(
        self,
        flattened_frame: Mapping[str, Any],
    ) -> None:
        if self._episode_stack_mode is None:
            self._episode_stack_mode = "mapping"
            self.episodic_frames = {
                key: [value]
                for key, value in flattened_frame.items()
            }
            return

        if self._episode_stack_mode != "mapping":
            raise ValueError(
                "Render frame type changed within an episode: expected non-mapping render frames, got mapping render frame."
            )

        assert isinstance(self.episodic_frames, dict)
        old_keys = set(self.episodic_frames.keys())
        new_keys = set(flattened_frame.keys())
        if old_keys != new_keys:
            missing_keys = sorted(old_keys - new_keys)
            added_keys = sorted(new_keys - old_keys)
            raise ValueError(
                f"Flattened render keys changed within an episode. Missing keys: {missing_keys}. New keys: {added_keys}."
            )

        for key in self.episodic_frames.keys():
            self.episodic_frames[key].append(flattened_frame[key])

    def _frame_to_numpy_rgb(
        self,
        frame: Any,
        key: Optional[str] = None,
    ) -> np.ndarray:
        """``frame`` as an (H, W, 3) RGB numpy image; ``key`` is its flattened key (``None`` for single frames)."""
        if self.env.backend.is_backendarray(frame):
            frame_np = self.env.backend.to_numpy(frame)
        else:
            assert isinstance(frame, np.ndarray), f"Expected backend array or np.ndarray, got {type(frame)}"
            frame_np = frame
        if self._image_specs is not None and key in self._image_specs:
            layout, format = self._image_specs[key]
            frame_np = image_to_hwc_rgb(frame_np, layout, format)
        assert frame_np.ndim == 3 and frame_np.shape[2] == 3, (
            f"Only RGB frames are supported, got shape {frame_np.shape}"
        )
        return frame_np

    def _frame_sequence_to_numpy_rgb(
        self,
        frames: Sequence[Any],
        key: Optional[str] = None,
    ) -> List[np.ndarray]:
        return [self._frame_to_numpy_rgb(frame, key) for frame in frames]

    def _is_image_frame(self, key: Optional[str], value: Any) -> bool:
        """Whether a (flattened, indexed) frame entry is recorded."""
        if self._image_specs is not None:
            return key in self._image_specs
        # No metadata: every render frame is an image; observations also hold non-image data
        return self.frame_source == "render" or _looks_like_rgb_image(value)

    def _transform_render_frame(
        self,
        frame: Any,
    ) -> Any:
        if isinstance(frame, Mapping):
            flattened_frame = flatten_nested_mapping(
                frame,
                nested_separator=self.nested_separator,
            )
            indexed = {
                key: self._index_render_tensor_if_needed(value)
                for key, value in flattened_frame.items()
            }
            if self._image_specs is None and self.frame_source == "render":
                return indexed
            return {key: value for key, value in indexed.items() if self._is_image_frame(key, value)}
        return self._index_render_tensor_if_needed(frame)

    def _capture_frame(self, observation: Any = None) -> Any:
        """The raw frame of this step: the observation or ``env.render()``, depending on ``frame_source``."""
        if self.frame_source == "observation":
            return observation
        return self.env.render()

    def _post_step_render(self, observation: Any = None) -> None:
        self._step_frame = self._transform_render_frame(self._capture_frame(observation))
        if isinstance(self._step_frame, Mapping):
            self._append_flattened_mapping_frame(self._step_frame)
        elif self._is_image_frame(None, self._step_frame):
            self._append_single_frame(self._step_frame)

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
                if self.done_index is None:
                    render_terminated = self.env.backend.any(mask)
                else:
                    render_terminated = bool(mask[self.done_index])

        if render_terminated:
            if self._episode_frame_count() > 1:
                if not self._has_post_episode:
                    self._post_episode()
                self.video_episode_num += 1
            
            self._reset_episode_stack()
            self._has_post_episode = False
            self.video_episode_start_step = self.video_step_num
        
        ret = self.env.reset(*args, mask=mask, seed=seed, **kwargs)
        self._post_step_render(ret[1])
        return ret
    
    def render(
        self
    ) -> Optional[Union[RenderFrame, Sequence[RenderFrame]]]:
        if self.frame_source != "render":
            return self.env.render()
        if self._step_frame is None:
            self._step_frame = self._transform_render_frame(self.env.render())
        return self._step_frame
    
    def close(self):
        if not self._has_post_episode:
            self._post_episode()
            self._has_post_episode = True
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
        done_index : Optional[int] = None,
        rendered_index : Optional[int] = None,
        nested_separator : str = ".",
        frame_source : Literal["render", "observation"] = "render",
        frame_space : Optional[Space] = None,
    ):
        super().__init__(
            env,
            done_index=done_index,
            rendered_index=rendered_index,
            nested_separator=nested_separator,
            frame_source=frame_source,
            frame_space=frame_space,
        )
        store_dir = os.path.abspath(store_dir)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        self.store_dir = store_dir
        self.store_format = format

    def _post_episode(self) -> None:
        if self._episode_frame_count() <= 1:
            return
        
        try:
           from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        except ImportError as e:
            raise ImportError(
                'MoviePy is not installed, run `pip install moviepy`'
            ) from e

        if self._episode_stack_mode == "mapping":
            assert isinstance(self.episodic_frames, dict)
            for mapping_key, frame_sequence in self.episodic_frames.items():
                video_path = os.path.join(
                    self.store_dir,
                    f"episode_{self.video_episode_num}_step_{self.video_episode_start_step}_key_{mapping_key}.{self.store_format}",
                )
                frames = self._frame_sequence_to_numpy_rgb(frame_sequence, mapping_key)
                self._write_clip(ImageSequenceClip, frames, video_path)
        else:
            assert isinstance(self.episodic_frames, list)
            video_path = os.path.join(
                self.store_dir,
                f"episode_{self.video_episode_num}_step_{self.video_episode_start_step}.{self.store_format}",
            )
            frames = self._frame_sequence_to_numpy_rgb(self.episodic_frames)
            self._write_clip(ImageSequenceClip, frames, video_path)

    def _write_clip(self, clip_cls, frames: List[np.ndarray], video_path: str) -> None:
        clip = clip_cls(frames, fps=self.env.render_fps or 30)
        try:
            if self.store_format == 'gif':
                # write_videofile has no codec for .gif files
                clip.write_gif(video_path, logger=None)
            else:
                clip.write_videofile(video_path, logger=None)
        finally:
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
        done_index : Optional[int] = None,
        rendered_index : Optional[int] = None,
        nested_separator : str = ".",
        frame_source : Literal["render", "observation"] = "render",
        frame_space : Optional[Space] = None,
    ):
        super().__init__(
            env,
            done_index=done_index,
            rendered_index=rendered_index,
            nested_separator=nested_separator,
            frame_source=frame_source,
            frame_space=frame_space,
        )
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

    def _frame_sequence_to_wandb_video(
        self,
        frame_sequence: Sequence[Any],
        key: Optional[str] = None,
    ):
        frames_np = np.stack(
            self._frame_sequence_to_numpy_rgb(frame_sequence, key),
            axis=0,
        )
        return self.wandb.Video(
            frames_np.transpose(0, 3, 1, 2),
            fps=self.env.render_fps or 30,
            format=self.store_format
        )

    def _post_episode(self) -> None:
        if self._episode_frame_count() <= 1:
            return

        to_log : Dict[str, Any] = {}
        if self._episode_stack_mode == "mapping":
            assert isinstance(self.episodic_frames, dict)
            for mapping_key, frame_sequence in self.episodic_frames.items():
                to_log[f"{self.wandb_log_key}/{mapping_key}"] = self._frame_sequence_to_wandb_video(frame_sequence, mapping_key)
        else:
            assert isinstance(self.episodic_frames, list)
            to_log[self.wandb_log_key] = self._frame_sequence_to_wandb_video(self.episodic_frames)

        if self.log_wandb_episode_id:
            to_log[self.wandb_log_key + '_episode_id'] = self.video_episode_num
        
        if self.control_wandb_step:
            self.wandb.log(to_log, step=self.video_episode_start_step)
        else:
            self.wandb.log(to_log, commit=False)
