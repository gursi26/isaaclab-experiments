import argparse

from omni.isaac.lab.app import AppLauncher
import torch
from utils import create_grid

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab_assets import CARTPOLE_CFG


# function to initialize the scene
def spawn_scene(sim_ctx):
	sim_ctx.set_camera_view(
		eye = [0.0, 0.0, 5.0],
		target = [0.0, 0.0, 0.0]
	)

	cfg_ground = sim_utils.GroundPlaneCfg()
	cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

	light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
	light_cfg.func("/World/Light", light_cfg)

	scene_entities = {}
	origins = create_grid(5, 10.0)
	origins = origins.to(sim_ctx.device)

	for i, origin in enumerate(origins):
		prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

	cartpole_cfg = CARTPOLE_CFG.copy()
	cartpole_cfg.prim_path = "/World/Origin.*/Robot"
	cartpole = Articulation(cfg=cartpole_cfg)
	scene_entities["cartpole"] = cartpole

	return scene_entities, origins


# function that runs every k iterations to reset the object origins and rotations
def reset_objects(entities, origins):
	print("[INFO]: Resetting object state")
	cp = entities["cartpole"]

	# setting root state
	root_state = cp.data.default_root_state.clone()
	root_state[:, :3] += origins
	cp.write_root_state_to_sim(root_state)

	# setting random pole angle
	pos, vel = cp.data.default_joint_pos.clone(), cp.data.default_joint_vel.clone()
	pos = torch.rand_like(pos) * 0.1
	cp.write_joint_state_to_sim(pos, vel)
	cp.reset()


# simulator loop
def run_simulator(sim, entities, origins):
	sim.reset()
	sim_step = 0
	sim_time = 0.0
	sim_dt = sim.get_physics_dt()
	cp = entities["cartpole"]

	while simulation_app.is_running():
		if sim_step % 600 == 0:
			reset_objects(entities, origins)

		random_movement = torch.randn_like(cp.data.joint_pos) * 5.0
		cp.set_joint_effort_target(random_movement)
		cp.write_data_to_sim()

		sim.step()
		sim_step += 1
		sim_time += sim_dt
		cp.update(sim_dt)


if __name__ == "__main__":
	sim_cfg = SimulationCfg(dt = 1 / 60)
	sim = SimulationContext(sim_cfg)
	entities, origins = spawn_scene(sim)
	run_simulator(sim, entities, origins)
	simulation_app.close()