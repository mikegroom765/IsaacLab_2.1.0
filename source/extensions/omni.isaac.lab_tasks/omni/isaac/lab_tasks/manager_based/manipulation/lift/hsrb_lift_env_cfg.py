# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.sensors import RayCasterCfg, TiledCameraCfg, ContactSensorCfg

from . import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach import mdp as reach_mdp
from omni.isaac.lab_tasks.manager_based.manipulation.inhand import mdp as in_hand_mdp


from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##

illegal_termination_list = ["base_link_contact", "base_contact_back", "base_contact_front"] # , #, 
                            # "torso_lift_link_contact", "arm_contact_lift", "arm_contact_flex"] 
                            # "arm_contact_roll"] #, "wrist_contact_roll"]
                            # "self_gripper_contact_hand_palm_link", 
                            # "self_gripper_contact_hand_l_spring_proximal_link", 
                            # "self_gripper_contact_hand_r_spring_proximal_link", 
                            # "self_gripper_contact_hand_l_distal_link", "self_gripper_contact_hand_r_distal_link", 
                            # "self_gripper_contact_hand_l_finger_vacuum_frame"]
                            # "head_pan_contact", "head_tilt_contact", "head_rgbd_sensor_contact"]
                            
penalty_contact_list = ["wrist_roll_contact_force",
                        "torso_lift_link_contact_force",
                        "arm_lift_contact_force",
                        "arm_flex_contact_force",
                        "arm_roll_contact_force",
                        "self_gripper_hand_palm_link_contact_force", 
                        "self_gripper_hand_l_spring_proximal_link_contact_force", 
                        "self_gripper_hand_r_spring_proximal_link_contact_force", 
                        "self_gripper_hand_l_distal_link_contact_force", 
                        "self_gripper_hand_r_distal_link_contact_force", 
                        "self_gripper_hand_l_finger_vacuum_frame_contact_force"]
                            
num_steps_per_env = 48

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING
    
    terrain: TerrainImporterCfg = MISSING

    height_scan: RayCasterCfg | None = None
    depth_camera_tiled: TiledCameraCfg | None = None
    lidar: RayCasterCfg | None = None
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
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
    left_gripper_object_contact_force: ContactSensorCfg | None = None
    right_gripper_object_contact_force: ContactSensorCfg | None = None
    
    # ee_tcp_debug = FrameTransformerCfg(
    #         prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
    #         debug_vis=True,
    #         visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EETCPFrameTransformer"),
    #         target_frames=[
    #             FrameTransformerCfg.FrameCfg(
    #                 prim_path="{ENV_REGEX_NS}/Robot/ee_tcp",
    #                 name="depth_camera",
    #                 offset=OffsetCfg(
    #                     pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
    #                 ),
    #             ),
    #         ],
    #     )



##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_goal_region: mdp.GoalRegionCommandCfg | None = None
    risk_sensitivity: mdp.ScalarValueCommandCfg = mdp.ScalarValueCommandCfg(
            value_range=(-1.0, 1.0),
            resampling_time_range=(1000.0, 1000.0),
        )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: ActionTerm | None = None
    base_action: ActionTerm | None = None
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_pos: ObsTerm | None = None
        joint_pos: ObsTerm | None = None
        base_vel: ObsTerm | None = None
        joint_vel: ObsTerm | None = None
        risk_sensitivity: ObsTerm | None = ObsTerm(func=mdp.generated_commands, params={"command_name": "risk_sensitivity"})
        current_ee_pose_base_frame = ObsTerm(func=reach_mdp.ee_pose_robot_base_frame, params={"make_quat_unique": True}, noise=Unoise(n_min=-0.01, n_max=0.01))
        masked_object_position: ObsTerm | None = None
        # initial_object_position = ObsTerm(func=mdp.initial_object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.pose_command_base_frame, params={"command_name": "object_goal_region"})
        is_grasped_object = ObsTerm(func=mdp.is_grasping_object_observation)
        ee_distance_to_object = ObsTerm(func=mdp.ee_distance_to_object)
        object_distance_to_goal = ObsTerm(func=mdp.object_distance_to_goal,  params={"command_name": "object_goal_region"})
        actions = ObsTerm(func=mdp.last_action)
        
        lidar_scan: ObsTerm | None = None
        depth_image: ObsTerm | None = None
            
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

        # rel_cylinder_pos: ObsTerm | None = None # relative position of the cylinder to the robot base

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    depth: ObsGroup | None = None
    height_scan: HeightScanCfg = HeightScanCfg()
    critic_policy: CriticPolicyCfg = CriticPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints: EventTerm | None = EventTerm(
            func=mdp.reset_robot_joints_set_table_height,
            mode="reset",
            params={
                # "position_range": (-0.0, 0.0),
                "position_range": (-0.5, 0.5),
                "velocity_range": (0.0, 0.0),
                "table_heights": MISSING,
            },
        )

    reset_base: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform_per_env,
        mode="reset",
        params={
            "initial_root_state": {"x": (0.57, 0.57), "y": (0.04, 0.04), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "initial_velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "x_pose_range": (0.48, 0.0),
        },
    )    
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform_set_height_per_env,
        mode="reset",
        params={
            "initial_position_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
            "position_range": {"x": (0.0, 0.25), "y": (0.0, 0.25)},
            "set_heights": MISSING,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
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
    
    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_reward = RewTerm(func=mdp.object_ee_distance_bodies, params={"std": 0.2}, weight=1.0)
    
    is_grasped_object = RewTerm(func=mdp.is_grasping_object, weight=5.0)
    
    object_goal_distance_when_grasped = RewTerm(
        func=mdp.object_goal_distance_when_grasped,
        params={"command_name": "object_goal_region",
                "grasped_reward_term_name": "is_grasped_object",
                "std": 1.0,
                "min_force": 0.5,
                "min_angle": 85,},
        weight=5.0,
    )
    
    static_vel_reward_when_placed = RewTerm(
        func=mdp.static_vel_reward_when_placed,
        params={"command_name": "object_goal_region",
                "arm_joint_names": ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
                "object_goal_distance_threshold": 0.25,
                "std": 3},
        weight=10.0,
    )

    success_reward = RewTerm(
        func=mdp.success_reward,
        params={"command_name": "object_goal_region",
                "arm_joint_names": ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
                "object_goal_distance_threshold": 0.15,
                "robot_joint_vel_threshold": 1.5},
        weight=20.0,
    )

    # action penalty
    # TODO: check logs for what the weights where last time
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    
    arm_joint_vel = RewTerm(
        func=mdp.joint_vel_l2_filtered,
        weight=-0.03,
        params={"joint_names": ["arm_lift_joint",
                                "arm_flex_joint",
                                "arm_roll_joint",
                                "wrist_flex_joint",
                                "wrist_roll_joint"], 
                "asset_cfg": SceneEntityCfg("robot")},
    )
    
    base_vel = RewTerm(
        func=mdp.body_lin_vel_l2_filtered,
        weight=-0.01,
        params={"body_names": ["base_link"],
                "asset_cfg": SceneEntityCfg("robot")},
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-20.0,
        params={"term_keys": illegal_termination_list},
    )
    
    # contact penalty
    undesired_contacts_penalty = RewTerm(
        func=mdp.undesired_contacts_list,
        weight=-0.2,
        params={"threshold": 0.1, "sensor_cfg_list": penalty_contact_list},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimums, params={"minimum_heights": MISSING, "threshold": 0.05, "asset_cfg": SceneEntityCfg("object")}
    # )
    
    base_link_contact = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_link_contact_force")})
    base_contact_back = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_b_bumper_contact_force", body_names=[".*"])})
    base_contact_front = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_f_bumper_contact_force", body_names=[".*"])})
    # torso_lift_link_contact = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("torso_lift_link_contact_force")})
    # arm_contact_lift = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_lift_contact_force")})
    # arm_contact_flex = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_flex_contact_force")})
    # arm_contact_roll = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_roll_contact_force")})
    # wrist_contact_roll = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("wrist_roll_contact_force")})
    # self_gripper_contact_hand_palm_link = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_palm_link_contact_force")})
    # self_gripper_contact_hand_l_spring_proximal_link = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_spring_proximal_link_contact_force")})
    # self_gripper_contact_hand_r_spring_proximal_link = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_spring_proximal_link_contact_force")})
    # self_gripper_contact_hand_l_distal_link = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_distal_link_contact_force")})
    # self_gripper_contact_hand_r_distal_link = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_distal_link_contact_force")})
    # self_gripper_contact_hand_l_finger_vacuum_frame = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_finger_vacuum_frame_contact_force")})
    # head_pan_contact = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("head_pan_contact_force")})
    # head_tilt_contact = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("head_tilt_contact_force")})
    # head_rgbd_sensor_contact = DoneTerm(func=mdp.filtered_illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("head_rgbd_sensor_contact_force")})
    object_out_of_reach = DoneTerm(
        func=in_hand_mdp.object_away_from_goal, params={"threshold": 1.0, "command_name": "object_goal_region"}
    )
    goal_reached: DoneTerm | None = None


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 10000}
    # )

    arm_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm_joint_vel", "weight": -0.1, "num_steps": 300 * num_steps_per_env}
    )
    
    levels = CurrTerm(
        func=reach_mdp.lift_cube_curriculum, 
        params={"event_terms": ["reset_base", "reset_object_position", "reset_robot_joints"],
                "level_intervals": 5,
                "successful_envs_term": "is_grasped_object"}
    )


##
# Environment configuration
##


@configclass
class HSRBLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.decimation = 4 # -> 30Hz control
        self.episode_length_s = 6.0
        # simulation settings
        self.sim.dt = 1.0 / 120 # -> 120Hz physics
        self.sim.render_interval = 12 # -> 10Hz rendering
        self.is_finite_horizon = False

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
