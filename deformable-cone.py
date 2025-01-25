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
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg

# function to initialize the scene
def spawn_scene(sim_ctx):
	sim_ctx.set_camera_view(
		eye = [0.0, 0.0, 5.0],		# eye location in 3d
		target = [0.0, 0.0, 0.0]	# location in 3d where eye vector points
	)

	cfg_ground = sim_utils.GroundPlaneCfg()
	cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

	light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
	light_cfg.func("/World/Light", light_cfg)

	# spawn multiple groups, each with robot and object (parallel training)
	origins = create_grid(10, spacing=3.0)
	for i, origin in enumerate(origins):
		prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

	# Rigid Object
	cone_cfg = DeformableObjectCfg(
		prim_path="/World/Origin.*/Cone", 	# regex path to spawn cone in all groups
		spawn=sim_utils.MeshConeCfg(			# ConeCfg is wrapped inside RigidObjectCfg
			radius=0.5,
			height=1.0,
			deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
			physics_material=sim_utils.DeformableBodyMaterialCfg(
				poissons_ratio=0.4,
				youngs_modulus=1e5
			),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
			# mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
			# collision_props=sim_utils.CollisionPropertiesCfg(),
		),
		init_state=DeformableObjectCfg.InitialStateCfg(),
	)
	cone_object = DeformableObject(cfg=cone_cfg)
	scene_entities = {"cone": cone_object}
	return scene_entities, origins

# function that runs every k iterations to reset the object origins and rotations
def reset_objects(entities, origins):
	nodal_state = entities["cone"].data.default_nodal_state_w.clone()
	N_objects = nodal_state.shape[0]

	# writing origins. root state is absolute coords so we offset from the origin
	pos_w = torch.rand(N_objects, 3, device="cuda") * 0.1 + origins
	pos_w[:, -1] += 5.0
	quat_w = math_utils.random_orientation(N_objects, device="cuda")
	nodal_state[..., :3] = entities["cone"].transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

	entities["cone"].write_nodal_state_to_sim(nodal_state)
	entities["cone"].reset()
	print("[INFO]: Resetting object state")

# simulator loop
def run_simulator(sim, entities, origins):
	sim.reset()
	sim_step = 0
	sim_time = 0.0
	sim_dt = sim.get_physics_dt()
	origins = origins.to("cuda")

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