# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.sensors import FrameTransformerCfg, RayCasterCfg, TiledCameraCfg, ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab.terrains.config.hsrb_reach import HSRB_REACH_TERRAINS_CFG, HSRB_REACH_CORRIDOR_TERRAINS_CFG  # isort: skip



# import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from . import mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp


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
        terrain_generator=HSRB_REACH_TERRAINS_CFG, # HSRB_REACH_CORRIDOR_TERRAINS_CFG
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

    # moving cylinder
    cylinder: ArticulationCfg | None = None

    # objects
    object_1: RigidObjectCfg | None = None
    object_2: RigidObjectCfg | None = None
    object_3: RigidObjectCfg | None = None
    object_4: RigidObjectCfg | None = None
    object_5: RigidObjectCfg | None = None

    # lights
    sky_light: AssetBaseCfg | None = None 

    # sensors
    lidar: RayCasterCfg | None = None
    depth_camera: RayCasterCfg | None = None
    depth_camera_tiled: TiledCameraCfg | None = None

    # priviledged sensors
    scandots: RayCasterCfg | None = None
    height_scan: RayCasterCfg | None = None

    # contact sensors
    base_link_contact_force: ContactSensorCfg | None = None
    base_b_bumper_contact_force: ContactSensorCfg | None = None
    base_f_bumper_contact_force: ContactSensorCfg | None = None
    torso_lift_link_contact_force: ContactSensorCfg | None = None
    arm_lift_contact_force: ContactSensorCfg | None = None
    arm_flex_contact_force: ContactSensorCfg | None = None
    arm_roll_contact_force: ContactSensorCfg | None = None
    wrist_roll_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_palm_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_l_spring_proximal_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_r_spring_proximal_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_l_distal_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_r_distal_link_contact_force: ContactSensorCfg | None = None
    self_gripper_hand_l_finger_vacuum_frame_contact_force: ContactSensorCfg | None = None
    head_pan_contact_force: ContactSensorCfg | None = None
    head_tilt_contact_force: ContactSensorCfg | None = None
    head_rgbd_sensor_contact_force: ContactSensorCfg | None = None
    gripper_contact_force: ContactSensorCfg | None = None
    cylinder_contact_force: ContactSensorCfg | None = None
    cylinder_world_contact_force: ContactSensorCfg | None = None

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose: CommandTermCfg | None = None

    risk_sensitivity: mdp.ScalarValueCommandCfg | None = mdp.ScalarValueCommandCfg(
        value_range=(-1.5, 1.5),
        resampling_time_range=(10.0, 30.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm | None = None
    base_action: ActionTerm | None = None
    gripper_action: ActionTerm | None = None
    give_up_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pos: ObsTerm | None = None
        joint_pos: ObsTerm | None = None
        base_vel: ObsTerm | None = None
        joint_vel: ObsTerm | None = None
        ee_pose_command_base_frame = ObsTerm(func=mdp.pose_command_base_frame, params={"command_name": "ee_pose"})
        current_ee_pose_base_frame = ObsTerm(func=mdp.ee_pose_robot_base_frame, params={"make_quat_unique": True}, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        # if anymore observations are added, add them below risk_sensitivity, otherwise you will need to change 
        # the value_measure_adaptation term in the rsl_rl_cfg.py file (HSRBReachDPPOMultiInputRunnerCfg)
        risk_sensitivity: ObsTerm | None = ObsTerm(func=mdp.generated_commands, params={"command_name": "risk_sensitivity"})
        distance_to_goal = ObsTerm(func=mdp.distance_to_goal, params={"command_name": "ee_pose"})
        rel_cylinder_pos: ObsTerm | None = None # relative position of the cylinder to the robot base

        # TODO: Add proper noise to the observations for domain randomization

        # Define terms for the sensors - these should be instantiated in the task sub class!
        lidar_scan: ObsTerm | None = None
        depth_camera: ObsTerm | None = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class HeightScanCfg(ObsGroup):
        """Observations for height scan group."""

        height_scan: ObsTerm | None = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticPolicyCfg(PolicyCfg):
        """Critic policy observations. Same as policy observations, without noise."""

        rel_cylinder_pos: ObsTerm | None = None # relative position of the cylinder to the robot base

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    depth: ObsGroup | None = None
    height_scan: HeightScanCfg | None = HeightScanCfg()
    critic_policy: CriticPolicyCfg = CriticPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints: EventTerm | None = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            # "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )
    
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.75, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    robot_physics_material_num_rollouts = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="num_rollouts",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping_num_rollouts = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="num_rollouts",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.75, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    reset_gravity: EventTerm | None = None
    reset_joint_effort_limits: EventTerm | None = None

    reset_cylinder: EventTerm | None = None
    resample_cylinder_velocity: EventTerm | None = None

    reset_object_1: EventTerm | None = None
    reset_object_2: EventTerm | None = None
    reset_object_3: EventTerm | None = None
    reset_object_4: EventTerm | None = None
    reset_object_5: EventTerm | None = None

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    # end_effector_position_tracking = RewTerm(
    #     func=mdp.position_command_error_frame_shaped,
    #     weight=5.0,
    #     params={"frame_name": "ee_frame", "command_name": "ee_pose", "position_threshold": 0.30, "goal_reward": 20.0},
    # )
    # end_effector_position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh_frame,
    #     weight=5.0,
    #     params={"frame_name": "ee_frame", "command_name": "ee_pose", "std": 0.1},
    # )
    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error_frame_shaped,
    #     weight=0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "ee_pose", "frame_name": "ee_frame", "orientation_threshold": math.pi/4},
    # )
    # end_effector_height_tracking = RewTerm(
    #     func=mdp.height_command_error_frame,
    #     weight=-0.25,
    #     params={"frame_name": "ee_frame", "command_name": "ee_pose"},
    # )

    # # action rate penalty
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0005)
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_limits,
    #     weight=-0.001,
    #     params={"soft_ratio": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    # )
    # joint_acc = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     weight=-0.0001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # ee_acc = RewTerm(
    #     func=mdp.body_lin_acc_l2_filtered,
    #     weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "name_key": "hand_palm_link"},
    # )

    # # Stay alive penalty - encourage the agent to acomplish the task quickly
    alive = RewTerm(func=mdp.is_alive, weight=-0.05)
    # # Terminatation penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)

    ## Gripper Reward Terms ##
    ## Note: These are not needed for the reach task, but are included for future use elsewhere!
    # reward for closing the gripper when near the command pose
    # distance_threshold: float, orientation_threshold: float, command_name: str, frame_name:str, open_joint_pos: float, asset_cfg: SceneEntityCfg
    # gripper_close_reward = RewTerm(
    #     func=mdp.grasp_close,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), 
    #             "command_name": "ee_pose", 
    #             "frame_name": "ee_frame", 
    #             "open_joint_pos": MISSING, 
    #             "distance_threshold": MISSING, 
    #             "orientation_threshold": MISSING},
    # )

    # # is goal in camera view - use when using tiled camera
    # is_goal_in_camera_view = RewTerm(
    #     func=mdp.is_goal_in_camera_view,
    #     weight=0.1,
    #     params={"camera_name": "depth_camera_tiled", "goal_name": "ee_pose"}, 
    # )

    # # is goal in camera view - use when not using tiled camera
    # is_goal_in_camera_view = RewTerm(
    #     func=mdp.is_goal_in_camera_view_frame,
    #     weight=0.25,
    #     params={"camera_frame_name": "depth_camera_frame", "goal_name": "ee_pose", "camera_intrinsics": MISSING}, 
    # )

    #### Contact Force Penalties (Velocity scaled) ####
    #### Base ####
    # contact_penalty_base_b_bumper_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=2.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_b_bumper_contact_force", body_names=[".*"]), "asset_cfg": SceneEntityCfg("robot"), "link_name": "base_link"},
    # )
    # contact_penalty_base_f_bumper_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=2.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_f_bumper_contact_force", body_names=[".*"]), "asset_cfg": SceneEntityCfg("robot"), "link_name": "base_link"},
    # )
    #### Arm ####
    # contact_penalty_arm_flex_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_flex_contact_force"), "asset_cfg": SceneEntityCfg("robot"), "link_name": "arm_flex_link"},
    # )
    # contact_penalty_arm_roll_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_roll_contact_force"), "asset_cfg": SceneEntityCfg("robot"), "link_name": "arm_roll_link"},
    # )
    # #### Wrist ####
    # contact_penalty_wrist_roll_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("wrist_roll_contact_force"), "asset_cfg": SceneEntityCfg("robot"), "link_name": "wrist_roll_link"},
    # )
    #### (Self-collisions) Gripper ####
    # contact_penalty_self_gripper_hand_palm_link_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_palm_link_contact_force"), "asset_cfg": SceneEntityCfg("robot"), "link_name": "hand_palm_link"},
    # )
    # contact_penalty_self_gripper_hand_l_spring_proximal_link_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_spring_proximal_link_contact_force"), "asset_cfg": SceneEntityCfg("robot"), "link_name": "hand_l_spring_proximal_link"},
    # )
    # contact_penalty_self_gripper_hand_r_spring_proximal_link_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_spring_proximal_link_contact_force"), "asset_cfg": SceneEntityCfg("robot"), "link_name": "hand_r_spring_proximal_link"},
    # )
    # contact_penalty_self_gripper_hand_l_distal_link_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_distal_link_contact_force"), "asset_cfg": SceneEntityCfg("robot"), "link_name": "hand_l_distal_link"},
    # )
    # contact_penalty_self_gripper_hand_r_distal_link_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_distal_link_contact_force"), "asset_cfg": SceneEntityCfg("robot"), "link_name": "hand_r_distal_link"},
    # )
    # contact_penalty_self_gripper_hand_l_finger_vacuum_frame_velocity_scaled = RewTerm(
    #     func=mdp.contact_penalty_velocity_scaled,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_finger_vacuum_frame_contact_force"), "asset_cfg": SceneEntityCfg("robot"), "link_name": "hand_l_finger_vacuum_frame"},
    # )

    #### Contact Force Penalties ####
    #### Base ####
    # contact_penalty_base_b_bumper = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=2.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_b_bumper_contact_force", body_names=[".*"])},
    # )
    # contact_penalty_base_f_bumper = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=2.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_f_bumper_contact_force", body_names=[".*"])},
    # )
    #### Arm ####
    # contact_penalty_arm_flex = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_flex_contact_force")},
    # )
    # contact_penalty_arm_roll = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_roll_contact_force")},
    # )
    # #### Wrist ####
    # contact_penalty_wrist_roll = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("wrist_roll_contact_force")},
    # )
    #### (Self-collisions) Gripper ####
    # contact_penalty_self_gripper_hand_palm_link = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_palm_link_contact_force")},
    # )
    # contact_penalty_self_gripper_hand_l_spring_proximal_link = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_spring_proximal_link_contact_force")},
    # )
    # contact_penalty_self_gripper_hand_r_spring_proximal_link = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_spring_proximal_link_contact_force")},
    # )
    # contact_penalty_self_gripper_hand_l_distal_link = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_distal_link_contact_force")},
    # )
    # contact_penalty_self_gripper_hand_r_distal_link = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_distal_link_contact_force")},
    # )
    # contact_penalty_self_gripper_hand_l_finger_vacuum_frame = RewTerm(
    #     func=mdp.contact_penalty,
    #     weight=1.0,
    #     params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_finger_vacuum_frame_contact_force")},
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact_back = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_b_bumper_contact_force", body_names=[".*"])})
    # base_contact_front = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_f_bumper_contact_force", body_names=[".*"])})
    # arm_contact_flex = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_flex_contact_force")})
    # arm_contact_roll = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_roll_contact_force")})
    # wrist_contact_roll = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("wrist_roll_contact_force")})
    # self_gripper_contact_hand_palm_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_palm_link_contact_force")})
    # self_gripper_contact_hand_l_spring_proximal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_spring_proximal_link_contact_force")})
    # self_gripper_contact_hand_r_spring_proximal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_spring_proximal_link_contact_force")})
    # self_gripper_contact_hand_l_distal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_distal_link_contact_force")})
    # self_gripper_contact_hand_r_distal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_distal_link_contact_force")})
    # self_gripper_contact_hand_l_finger_vacuum_frame = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_finger_vacuum_frame_contact_force")})
    # goal_reached = DoneTerm(func=mdp.command_resample, params={"command_name": "ee_pose", "num_resamples": 2})# , time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    ### 2000 steps triggers during 22nd iteration - I think this is the start of the 3rd episode ###
    ### num_steps is calculated as total_steps_per_env = current_iteration * self._num_steps_per_env ###

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 19200} # Update after 200 iterations
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.005, "num_steps": 19200}
    # )

    # end_effector_position_tracking = CurrTerm(
    #     func=mdp.modify_reward_parameters, params={"term_name": "end_effector_position_tracking", "parameters": {"frame_name": "ee_frame", "command_name": "ee_pose", "position_threshold": 0.10}, "num_steps": 72000} # Update after 750 iterations
    # )

    # end_effector_orientation_tracking = CurrTerm(
    #     func=mdp.modify_reward_parameters, params={"term_name": "end_effector_orientation_tracking", "parameters": {"asset_cfg": SceneEntityCfg("robot"), "command_name": "ee_pose", "frame_name": "ee_frame", "orientation_threshold": math.pi/8}, "num_steps": 72000}
    # )
    
    # # ee_pose_command_range = CurrTerm(
    #     # func=mdp.modify_command_parameters, params={"command_name": "ee_pose", "parameters": {"pos_x": (-2.0, 2.0), "pos_y": (-2.0, 2.0), "pos_z": (0.2, 1.4)}, "num_steps": 72000}
    # # )

    # termination_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "termination_penalty", "weight": -50.0, "num_steps": 108000} # 1125 iterations
    # )

    # ee_acc = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "ee_acc", "weight": -0.005, "num_steps": 144000} # 1500 iterations
    # )

    # # terrain_levels = CurrTerm(func=mdp.terrain_levels_pos_ori, params={"position_thresholds": [0.5, 2.5], "orientation_thresholds": [math.pi/1.5, 1.5*math.pi]})
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_successful_envs)

    goal_levels: CurrTerm | None = None
    goal_levels_range: CurrTerm | None = None

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
    # MDP settingsprint(f"Command name: {command_name}, Command value: {env.command_manager.get_command(command_name)[0]}")tionsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4 # -> 30hz control
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 120.0 # -> 120hz physics 

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
        self.decimation = 4 # -> 30hz control
        self.sim.render_interval = 12 # -> 10hz rendering
        self.episode_length_s = 30.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 120.0 # -> 120hz physics
        # set is_finite_horizon to False
        self.is_finite_horizon = False

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False