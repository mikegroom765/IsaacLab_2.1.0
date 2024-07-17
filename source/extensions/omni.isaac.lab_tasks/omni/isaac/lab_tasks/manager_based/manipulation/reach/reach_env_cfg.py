# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.sensors import FrameTransformerCfg, RayCasterCfg, TiledCameraCfg, ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab.terrains.config.hsrb_reach import HSRB_REACH_TERRAINS_CFG  # isort: skip



# import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from . import mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

@configclass
class HSRBReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a hsrb."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HSRB_REACH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # frame transformers
    ee_frame: FrameTransformerCfg = MISSING
    depth_camera_frame: FrameTransformerCfg = MISSING

    # sensors
    lidar: RayCasterCfg | None = None
    depth_camera: RayCasterCfg | None = None
    depth_camera_tiled: TiledCameraCfg | None = None

    # contact sensors
    base_b_bumper_contact_force: ContactSensorCfg | None = None
    base_f_bumper_contact_force: ContactSensorCfg | None = None
    arm_flex_contact_force: ContactSensorCfg | None = None
    arm_roll_contact_force: ContactSensorCfg | None = None
    wrist_roll_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_palm_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_l_spring_proximal_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_r_spring_proximal_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_l_distal_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_r_distal_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_l_finger_vacuum_frame_contact_force: ContactSensorCfg | None = None
    gripper_contact_force: ContactSensorCfg | None = None

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(10.0, 20.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.2, 1.4),
            roll=(0.0, 0.0),
            pitch=MISSING,  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    base_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        current_ee_pos = ObsTerm(func=mdp.ee_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)

        # TODO: Add global pose (odom) to the observation space - I'm pretty sure this is joint_pos[0:2] since the robot uses dummy joints?

        lidar_scan = ObsTerm(
            func=mdp.lidar_2d_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
        )
        # depth_camera = ObsTerm(
        #     func=mdp.depth_camera,
        #     params={"sensor_cfg": SceneEntityCfg("depth_camera"), "asset_cfg": SceneEntityCfg("robot")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     # clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DepthCfg(ObsGroup):
        """Observations for policy group."""

        depth_camera = ObsTerm(
            func=mdp.depth_camera,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    depth: DepthCfg = DepthCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_frame,
        weight=-1.0,
        params={"frame_name": "ee_frame", "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh_frame,
        weight=1.0,
        params={"frame_name": "ee_frame", "command_name": "ee_pose", "std": 0.1},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_frame,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "ee_pose", "frame_name": "ee_frame"},
    )
    end_effector_height_tracking = RewTerm(
        func=mdp.height_command_error_frame,
        weight=-1.0,
        params={"frame_name": "ee_frame", "command_name": "ee_pose"},
    )

    # action rate penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # Terminatation penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)

    # reward for closing the gripper when near the command pose
    # distance_threshold: float, orientation_threshold: float, command_name: str, frame_name:str, open_joint_pos: float, asset_cfg: SceneEntityCfg
    gripper_close_reward = RewTerm(
        func=mdp.grasp_close,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), 
                "command_name": "ee_pose", 
                "frame_name": "ee_frame", 
                "open_joint_pos": MISSING, 
                "distance_threshold": MISSING, 
                "orientation_threshold": MISSING},
    )

    # is goal in camera view - use when using tiled camera
    is_goal_in_camera_view = RewTerm(
        func=mdp.is_goal_in_camera_view,
        weight=0.1,
        params={"camera_name": "depth_camera_tiled", "goal_name": "ee_pose"}, 
    )

    # # is goal in camera view - use when not using tiled camera
    # is_goal_in_camera_view = RewTerm(
    #     func=mdp.is_goal_in_camera_view_frame,
    #     weight=0.1,
    #     params={"camera_name": "depth_camera", "goal_name": "ee_pose", "camera_intrinsics": MISSING}, 
    # )

    #### Contact Force Penalties ####
    #### Base ####
    contact_penalty_base_b_bumper = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_b_bumper_contact_force", body_names=[".*"])},
    )
    contact_penalty_base_f_bumper = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_f_bumper_contact_force", body_names=[".*"])},
    )
    #### Arm ####
    contact_penalty_arm_flex = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_flex_contact_force")},
    )
    contact_penalty_arm_roll = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_roll_contact_force")},
    )
    #### Wrist ####
    contact_penalty_wrist_roll = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("wrist_roll_contact_force")},
    )
    #### (Self-collisions) Gripper ####
    contact_penalty_self_gripper_hand_palm_link = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_palm_link_contact_force")},
    )
    contact_penalty_self_gripper_hand_l_spring_proximal_link = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_spring_proximal_link_contact_force")},
    )
    contact_penalty_self_gripper_hand_r_spring_proximal_link = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_spring_proximal_link_contact_force")},
    )
    contact_penalty_self_gripper_hand_l_distal_link = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_distal_link_contact_force")},
    )
    contact_penalty_self_gripper_hand_r_distal_link = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_distal_link_contact_force")},
    )
    contact_penalty_self_gripper_hand_l_finger_vacuum_frame = RewTerm(
        func=mdp.contact_penalty,
        weight=1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_finger_vacuum_frame_contact_force")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact_back = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_b_bumper_contact_force", body_names=[".*"])})
    base_contact_front = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_f_bumper_contact_force", body_names=[".*"])})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )

    terrain_levels = CurrTerm(func=mdp.terrain_levels_pos_ori)


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

@configclass
class MobileReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: HSRBReachSceneCfg = HSRBReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False