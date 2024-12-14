from omni.isaac.lab.envs import ManagerBasedEnvCfg
from omni.isaac.lab.utils import configclass
from scene import PlateBallSceneCfg

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
	scene = PlateBallSceneCfg(num_envs=1024, env_spacing=2.5)

	def __post_init__(self):
		"""Post initialization."""
		self.viewer.eye = [4.5, 0.0, 6.0]
		self.viewer.lookat = [0.0, 0.0, 2.0]
		self.decimation = 4 
		self.sim.dt = 0.005
