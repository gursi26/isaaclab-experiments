import argparse

from omni.isaac.lab.app import AppLauncher
import torch

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import mdp, ManagerBasedEnvCfg, ManagerBasedEnv
from omni.isaac.lab.managers import ObservationGroupCfg, ObservationTermCfg, EventTermCfg, SceneEntityCfg, RewardTermCfg, TerminationTermCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab_assets import CARTPOLE_CFG
import math

# specifies the scene for our task
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
	ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
	dome_light = AssetBaseCfg(
		prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
	)
	robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# contains specifications for the action space of environment
@configclass
class ActionsCfg:
	# efforts (forces) on the slider-cart joint are our actions
	joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


# contains specifications for the observation space of environment
@configclass
class ObservationsCfg:
	# we can define multiple observation groups to add structure to our observations
	@configclass
	class PolicyCfg(ObservationGroupCfg):
		# observation terms (in order)
		joint_pos_rel = ObservationTermCfg(func=mdp.joint_pos_rel)
		joint_vel_rel = ObservationTermCfg(func=mdp.joint_vel_rel)

		# ???
		def __post_init__(self) -> None:
			self.enable_corruption = False
			self.concatenate_terms = True

	# this observation group must be present
	policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
	"""Configuration for events."""

	# on startup
	add_pole_mass = EventTermCfg(
		func=mdp.randomize_rigid_body_mass,
		mode="startup",
		params={
			"asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
			"mass_distribution_params": (0.1, 0.5),
			"operation": "add",
		},
	)

	# on reset
	reset_cart_position = EventTermCfg(
		func=mdp.reset_joints_by_offset,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
			"position_range": (-1.0, 1.0),
			"velocity_range": (-0.1, 0.1),
		},
	)

	reset_pole_position = EventTermCfg(
		func=mdp.reset_joints_by_offset,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
			"position_range": (-0.125 * math.pi, 0.125 * math.pi),
			"velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
		},
	)

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewardTermCfg(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewardTermCfg(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewardTermCfg(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewardTermCfg(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = TerminationTermCfg(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
	# Scene settings
	scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
	# Basic settings
	observations = ObservationsCfg()
	actions = ActionsCfg()
	events = EventCfg()

	def __post_init__(self):
		"""Post initialization."""
		# viewer settings
		self.viewer.eye = [4.5, 0.0, 6.0]
		self.viewer.lookat = [0.0, 0.0, 2.0]
		# step settings
		self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
		# simulation settings
		self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
	env_cfg = CartpoleEnvCfg()
	env = ManagerBasedEnv(cfg=env_cfg)

	count = 0
	while simulation_app.is_running():
		with torch.inference_mode():
			if count % 300 == 0:
				count = 0
				env.reset()
				print("[INFO]: Resetting environment...")
			joint_efforts = torch.randn_like(env.action_manager.action)
			obs, _ = env.step(joint_efforts)
			print(f"actions: {env.action_manager.action.shape}")
			print(env.action_manager.action)
			print(f"observations: {obs['policy'].shape}")
			print(obs["policy"])
			count += 1

	# close the environment
	env.close()


if __name__ == "__main__":
	main()
	simulation_app.close()