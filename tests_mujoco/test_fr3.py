import pytest
import typing
from unienv_mujoco import MujocoFuncWorld, MujocoDefaultFuncActor, MujocoIKWrapper, MinkIK, MinkBulkIK, MujocoFuncWorldState, MujocoFuncWindowedViewSensor
from unienv_interface.space import *
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.env_base.funcenv import FuncEnv, StatefulSingleFuncEnv
from unienv_interface.world.world import FuncEnvCommonState
from unienv_interface.world.tasks import LambdaFuncTask
from unienv_interface.world.env_utils import WorldBasedFuncEnv
import unienv_mujoco.ik.ik_util as ik_util
import os.path
import mink
import numpy as np

CONTROL_TIMESTEP = 0.05
SUBSTEP_TIMESTEP = 0.005
AVOID_BODY_NAMES = ["fr3_link3", "fr3_link7"]
EEF_WORKSPACE = Box(
    backend=NumpyComputeBackend,
    low=np.array([0.2, -0.2, 0.2]),
    high=np.array([0.3, 0.2, 0.4]),
    dtype=np.float32
)
EEF_SE3_WORKSPACE = Box(
    backend=NumpyComputeBackend,
    low=np.concatenate(
        [EEF_WORKSPACE.low, np.array([-np.pi/2, -np.pi/2, -np.pi])],
    ),
    high=np.concatenate(
        [EEF_WORKSPACE.high, np.array([np.pi/2, np.pi/2, np.pi])],
    ),
    dtype=np.float32
)
STEP_LIMIT = 1_000

@pytest.fixture(scope="session")
def fr3_world() -> MujocoFuncWorld:
    _here = os.path.dirname(__file__)
    _xml = os.path.join(_here, "assets", "franka_fr3", "scene.xml")
    return MujocoFuncWorld(
        world_timestep=CONTROL_TIMESTEP,
        world_subtimestep=SUBSTEP_TIMESTEP,
        xml_path=_xml
    )

@pytest.fixture(scope="session")
def fr3_actor(fr3_world : MujocoFuncWorld) -> MujocoDefaultFuncActor:
    return MujocoDefaultFuncActor(
        world=fr3_world,
        control_timestep=CONTROL_TIMESTEP,
    )

@pytest.fixture(scope="session")
def fr3_eef_actor(fr3_world : MujocoFuncWorld, fr3_actor : MujocoDefaultFuncActor) -> MujocoIKWrapper:
    all_avoid_ids = []
    for body_name in AVOID_BODY_NAMES:
        body = fr3_world._mjmodel.body(body_name)
        all_avoid_ids.extend(mink.get_body_geom_ids(fr3_world._mjmodel, body.id))
    
    ik = MinkIK(
        collision_avoid_geom_pairs=[
            (all_avoid_ids, ["floor"]),
        ],
        # collision_avoid_geom_pairs=None,
        max_velocity_per_joint=None,
        frame_name="attachment_site",
        frame_type="site",
        relative_frame_name="actuation_center",
        relative_frame_type="site"
    )
    bulk_ik = MinkBulkIK(
        ik,
        additional_search_qpos=np.asarray([
            fr3_world._mjmodel.key("home").qpos
        ])
    )

    return MujocoIKWrapper(
        actor=fr3_actor,
        ik=bulk_ik,
        mj_model=fr3_world._mjmodel,
        new_action_space=EEF_SE3_WORKSPACE,
        fn_target_transform=lambda action: mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3.from_rpy_radians(*action[3:]),
            translation=action[:3]
        ),
        fn_action_transform=lambda action, target_q: target_q
    )

@pytest.fixture(scope="session")
def render_sensor(fr3_world : MujocoFuncWorld) -> MujocoFuncWindowedViewSensor:
    return MujocoFuncWindowedViewSensor(
        control_timestep=CONTROL_TIMESTEP,
        seed=None
    )


def eef_reward_and_termination_fn(
    target_transform : mink.SE3,
    eef_site_name : str = "attachment_site",
):
    def eef_reward_and_termination(
        world_state : MujocoFuncWorldState,
        common_state : FuncEnvCommonState[typing.Any, np.random.Generator],
        obs : typing.Dict[str, typing.Any],
        control_step_elapsed : float
    ):
        ik_transform = ik_util.get_transform_frame_to_world(
            world_state.mj_model,
            world_state.data,
            eef_site_name,
            "site"
        )
        diff = target_transform.minus(ik_transform)
        err_translation = np.linalg.norm(diff[:3])
        err_rotation = np.linalg.norm(diff[3:])
        if err_translation < 2e-2 and err_rotation < 1e-1:
            print("Achieved target!")
            return 1.0, True, False
        else:
            return 0.0, False, False
    return eef_reward_and_termination

def get_fr3_eef_task() -> typing.Tuple[LambdaFuncTask, np.ndarray]:
    target_eef = EEF_SE3_WORKSPACE.sample()
    assert target_eef.shape == (6,)
    assert EEF_SE3_WORKSPACE.contains(target_eef)
    print(f"Target EEF: {target_eef}")
    target_transform = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_rpy_radians(*target_eef[3:]),
        translation=target_eef[:3]
    )
    return LambdaFuncTask(
        None,
        None,
        eef_reward_and_termination_fn(target_transform)
    ), np.concatenate([target_transform.translation(), target_transform.rotation().as_rpy_radians()])

def test_fr3_eef(
    fr3_world : MujocoFuncWorld,
    fr3_eef_actor : MujocoIKWrapper,
    render_sensor : MujocoFuncWindowedViewSensor
):
    task, target_action = get_fr3_eef_task()
    target_transform = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_rpy_radians(*target_action[3:]),
        translation=target_action[:3]
    )

    funcenv = WorldBasedFuncEnv(
        world=fr3_world,
        actor=fr3_eef_actor,
        task=task,
        render_sensor=render_sensor,
        info_callback=None
    )
    env = StatefulSingleFuncEnv(
        funcenv,
        seed=0
    )
    env.reset()
    ik_util.move_mocap_to_transformation(
        env.state.world_state.mj_model,
        env.state.world_state.data,
        "target",
        target_transform
    )
    for _ in range(STEP_LIMIT):
        obs, rew, termination, truncation, info = env.step(target_action)
        if termination:
            break
    env.close()
    assert termination

def test_fr3_eef_interactive(
    fr3_world : MujocoFuncWorld,
    fr3_eef_actor : MujocoIKWrapper,
    render_sensor : MujocoFuncWindowedViewSensor
):
    task, target_action = get_fr3_eef_task()
    funcenv = WorldBasedFuncEnv(
        world=fr3_world,
        actor=fr3_eef_actor,
        task=task,
        render_sensor=render_sensor,
        info_callback=None
    )
    env = StatefulSingleFuncEnv(
        funcenv,
        seed=0
    )
    mink.move_mocap_to_frame(
        env.state.world_state.mj_model, 
        env.state.world_state.data, 
        "target", 
        "attachment_site", 
        "site"
    )
    env.reset()
    for _ in range(STEP_LIMIT):
        T_wt = mink.SE3.from_mocap_name(env.state.world_state.mj_model, env.state.world_state.data, "target")
        real_target = np.concatenate([T_wt.translation(), T_wt.rotation().as_rpy_radians()])
        obs, rew, termination, truncation, info = env.step(real_target)
        if termination:
            break
    env.close()
    assert termination
    
