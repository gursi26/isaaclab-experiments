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
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg


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
	origins = torch.zeros(1, 3, device=sim_ctx.device)

	#######################
	## PUT CODE HERE ##
	#######################

	return scene_entities, origins


# function that runs every k iterations to reset the object origins and rotations
def reset_objects(entities, origins):
	print("[INFO]: Resetting object state")
	#######################
	## PUT CODE HERE ##
	#######################


# simulator loop
def run_simulator(sim, entities, origins):
	sim.reset()
	sim_step = 0
	sim_time = 0.0
	sim_dt = sim.get_physics_dt()
	while simulation_app.is_running():
		if sim_step % 300 == 0:
			reset_objects(entities, origins)
		sim.step()
		sim_step += 1
		sim_time += sim_dt



if __name__ == "__main__":
	sim_cfg = SimulationCfg(dt = 1 / 60)
	sim = SimulationContext(sim_cfg)
	entities, origins = spawn_scene(sim)
	run_simulator(sim, entities, origins)
	simulation_app.close()