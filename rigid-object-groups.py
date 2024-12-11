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
		eye = [0.0, 0.0, 5.0],		# eye location in 3d
		target = [0.0, 0.0, 0.0]	# location in 3d where eye vector points
	)

	# initializes and spawns a ground plane
	cfg_ground = sim_utils.GroundPlaneCfg()
	cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

	# initializes and spawns a directional light
	cfg_light = sim_utils.DistantLightCfg(
		intensity = 3000.0,
		color = (0.75, 0.75, 0.75)
	)
	cfg_light.func("/World/defaultLight", cfg_light, translation=(100, 0, 10)) # location of light source


	# spawn multiple groups, each with robot and object (parallel training)
	origins = create_grid(100, spacing=3.0)
	for i, origin in enumerate(origins):
		prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

	# Rigid Object
	cone_cfg = RigidObjectCfg(
		prim_path="/World/Origin.*/Cone", 	# regex path to spawn cone in all groups
		spawn=sim_utils.ConeCfg(			# ConeCfg is wrapped inside RigidObjectCfg
			radius=0.1,
			height=0.2,
			rigid_props=sim_utils.RigidBodyPropertiesCfg(),
			mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(),
	)
	cone_object = RigidObject(cfg=cone_cfg)
	scene_entities = {"cone": cone_object}
	return scene_entities, origins

# function that runs every k iterations to reset the object origins and rotations
def reset_objects(entities, origins):
	root_state = entities["cone"].data.default_root_state.clone()
	N_objects = root_state.shape[0]

	# writing origins. root state is absolute coords so we offset from the origin
	root_state[:, :3] = origins
	root_state[:, :2] += (torch.rand((N_objects, 2), device="cuda") * 2) - 1.0
	root_state[:, 2] = (torch.rand(N_objects, device="cuda") * 3) + 1.0

	# writing rotation quaternions
	rand_rotation = torch.randn(N_objects, 4, device="cuda")
	rand_rotation = rand_rotation / rand_rotation.norm(dim=-1, keepdim=True)
	root_state[:, 3:7] = rand_rotation

	entities["cone"].write_root_state_to_sim(root_state)
	entities["cone"].reset()
	print("[INFO]: Resetting object state")

# simulator loop
def run_simulator(sim, entities, origins):
	sim.reset()
	sim_step = 0
	sim_time = 0.0
	sim_dt = sim.get_physics_dt()

	while simulation_app.is_running():
		if sim_step % 300 == 0:
			reset_objects(entities, origins)
		entities["cone"].write_data_to_sim()
		sim.step()
		sim_step += 1
		sim_time += sim_dt
		entities["cone"].update(sim_dt)


if __name__ == "__main__":
	sim_cfg = SimulationCfg(dt = 1 / 60)
	sim = SimulationContext(sim_cfg)
	entities, origins = spawn_scene(sim)
	run_simulator(sim, entities, origins)
	simulation_app.close()