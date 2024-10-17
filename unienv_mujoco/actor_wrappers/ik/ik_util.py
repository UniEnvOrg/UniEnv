import mink.constants as consts
import mink
import mujoco

def get_transform_frame_to_world(
    mj_model : mujoco.MjModel,
    mj_data : mujoco.MjData, 
    frame_name: str, 
    frame_type: str
) -> mink.SE3:
    """Get the pose of a frame at the current configuration.

    Args:
        frame_name: Name of the frame in the MJCF.
        frame_type: Type of frame. Can be a geom, a body or a site.

    Returns:
        The pose of the frame in the world frame.
    """
    if frame_type not in consts.SUPPORTED_FRAMES:
        raise NotImplementedError

    frame_id = mujoco.mj_name2id(
        mj_model, consts.FRAME_TO_ENUM[frame_type], frame_name
    )
    assert frame_id != -1, f"Invalid frame {frame_name} of type {frame_type}"

    xpos = getattr(mj_data, consts.FRAME_TO_POS_ATTR[frame_type])[frame_id]
    xmat = getattr(mj_data, consts.FRAME_TO_XMAT_ATTR[frame_type])[frame_id]
    return mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_matrix(xmat.reshape(3, 3)),
        translation=xpos,
    )

def get_relative_transform(
    self,
    base_transform : mink.SE3,
    query_transform : mink.SE3
) -> mink.SE3:
    return base_transform.inverse() @ query_transform

def get_absolute_transform(
    self,
    base_transform : mink.SE3,
    relative_transform : mink.SE3
) -> mink.SE3:
    return base_transform @ relative_transform
