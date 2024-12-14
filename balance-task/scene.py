import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg, RigidObject

def _get_plate_config():
	side_length = 1.0
	cfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Plate", 
		spawn=sim_utils.CuboidCfg(
			size=(side_length, side_length, 0.05),
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				disable_gravity=True,
				kinematic_enabled=True
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(0, 0, 2)
		),
	)
	return cfg


def _get_ball_config():
	radius = 0.1
	cfg = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Ball", 
		spawn=sim_utils.SphereCfg(
			radius=radius,
			rigid_props=sim_utils.RigidBodyPropertiesCfg(),
			mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(0, 0, 3)
		),
	)
	return cfg



@configclass
class PlateBallSceneCfg(InteractiveSceneCfg):
	ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
	dome_light = AssetBaseCfg(
		prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
	)
	plate: RigidObjectCfg = _get_plate_config()
	ball: RigidObjectCfg = _get_ball_config()