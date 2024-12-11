import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.sim import SimulationContext, SimulationCfg

def spawn_scene():
	# initializes and spawns a ground plane
	cfg_ground = sim_utils.GroundPlaneCfg()
	cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

	# initializes and spawns a directional light
	cfg_light = sim_utils.DistantLightCfg(
		intensity = 3000.0,
		color = (0.75, 0.75, 0.75)
	)
	cfg_light.func("/World/defaultLight", cfg_light, translation=(100, 0, 10)) # location of light source

	# a transform prim groups other objects so we can move the whole group together
	prim_utils.create_prim("/World/Objects", "Xform")

	# spawning a red cone
	cfg_cone = sim_utils.ConeCfg(
		radius = 0.15,
		height = 0.5,
		visual_material = sim_utils.PreviewSurfaceCfg(		# giving the cone surface properties
			diffuse_color=(1.0, 0.0, 0.0)
		),
		rigid_props = sim_utils.RigidBodyPropertiesCfg(),		# rigid body (no squishiness)
		mass_props = sim_utils.MassPropertiesCfg(mass=1.0),		# fixed mass for accurate physics simulations
		collision_props = sim_utils.CollisionPropertiesCfg(),	# collision box to prevent going through stuff
	)
	cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
	cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

	# spawn a blue cuboid with deformable body
	cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
		size=(0.2, 0.5, 0.2),
		deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
		visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
		physics_material=sim_utils.DeformableBodyMaterialCfg(),
	)
	cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

	# spawn from obj file
	cfg_bunny = sim_utils.UsdFileCfg(
		usd_path="assets/bunny.usd",
	)
	cfg_bunny.func("/World/Objects/Bunny", cfg_bunny, translation=(1.0, 0.0, 1.5))



if __name__ == "__main__":
	sim_cfg = SimulationCfg(
		dt = 1 / 60 				# dt - delta time in seconds between timesteps
	)
	sim = SimulationContext(sim_cfg)
	sim.set_camera_view(
		eye = [0.0, 0.0, 5.0],		# eye location in 3d
		target = [0.0, 0.0, 0.0]	# location in 3d where eye vector points
	)

	spawn_scene()
	sim.reset()
	print("[INFO]: Setup complete. Starting simulation...")
	while simulation_app.is_running():
		sim.step()

	simulation_app.close()
