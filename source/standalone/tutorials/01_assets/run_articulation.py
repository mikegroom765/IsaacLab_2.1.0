# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip
from omni.isaac.lab_assets.hsrb import HSRB_CFG  # isort:skip
from omni.isaac.lab_assets.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG  # isort:skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    # origins = [1.0, 0.0, 0.0]
    origins = []

    # create a n x n grid of origins, spaced 1 unit apart
    for i in range(10):
        for j in range(10):
            origins.append([i, j, 0.0])
            prim_utils.create_prim(f"/World/Origin{i}{j}", "Xform", translation=[i, j, 0.0])

    # Origin 1
    # prim_utils.create_prim("/World/Origin1", "Xform", translation=origins)
    # Origin 2
    # prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Articulation
    # cartpole_cfg = RIDGEBACK_FRANKA_PANDA_CFG.copy()
    cartpole_cfg = HSRB_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    # return the scene information
    scene_entities = {"cartpole": cartpole}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["cartpole"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    target_base_names = [
        "joint_x",
        "joint_y",
        "joint_rz"
    ]

    # target_base_names = [
    #     "dummy_base_prismatic_x_joint",
    #     "dummy_base_prismatic_y_joint",
    #     "dummy_base_revolute_z_joint"
    # ]

    target_base_index = [robot.data.joint_names.index(name) for name in target_base_names]

    actions = robot.data.default_joint_pos.clone()


    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 1000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.ones_like(robot.data.joint_pos) * 0.5

        # -- apply high efforts to joint_x, joint_y, joint_rz
        # efforts[:, 0] = 0.1
        # efforts[:, 1] = 0.1
        # efforts[:, 2] = 0.1
        if count == 0:
            # rotate motion
            actions[:, :3] = 0.0
            actions[:, 2] = 1.0
            print("rotate motion")
        if count == 200:
            #forward motion
            actions[:, :3] = 0.0
            actions[:, 0] = 1.0
            print("forward motion")
        elif count == 400:
            #backward motion
            actions[:, :3] = 0.0
            actions[:, 0] = -1.0
            print("backward motion")
        elif count == 600:
            #left motion
            actions[:, :3] = 0.0
            actions[:, 1] = 1.0
            print("left motion")
        elif count == 800:
            #right motion
            actions[:, :3] = 0.0
            actions[:, 1] = -1.0
            print("right motion")

        # print is_fixed_base
        # print("is_fixed_base: ", robot.is_fixed_base)

        # print joint_names
        # print("joint_names: ", robot.joint_names)

        # print("data: ", robot.data.joint_stiffness)
        
        # -- print size of joint_pos and joint_vel
        # print("joint_pos size: ", robot.data.joint_pos.size())
        # print("joint_vel size: ", robot.data.joint_vel.size())

        # -- apply action to the robot
        # print(target_base_index)
        # robot.set_joint_position_target(efforts)
        # robot.set_joint_position_target(actions[:, 3:], joint_ids=[3, 4, 5, 6, 7, 8, 9, 10, 11])
        robot.set_joint_velocity_target(actions[:, :3], joint_ids=target_base_index)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
