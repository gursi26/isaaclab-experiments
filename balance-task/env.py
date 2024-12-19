from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform, quat_from_euler_xyz, euler_xyz_from_quat

def _get_plate_config():
	side_length = 1.0
	cfg = RigidObjectCfg(
		prim_path="/World/envs/env_.*/Plate", 
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
		prim_path="/World/envs/env_.*/Ball", 
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
class PlateBallEnvCfg(DirectRLEnvCfg):
	num_envs = 64
	env_spacing = 2.0
	dt = 1 / 120
	observation_space = 100
	action_space = 1

	# env
	decimation = 2
	episode_length_s = 5.0

	# simulation
	sim: SimulationCfg = SimulationCfg(
		dt=dt, 
		render_interval=decimation
	)

	plate_cfg: RigidObjectCfg = _get_plate_config()
	ball_cfg: RigidObjectCfg = _get_ball_config()
	fall_threshold = 1.0
	action_damping = 0.02

	# scene
	scene: InteractiveSceneCfg = InteractiveSceneCfg(
		num_envs=num_envs, 
		env_spacing=env_spacing, 
		replicate_physics=True
	)


class PlateBallEnv(DirectRLEnv):
	cfg: PlateBallEnvCfg

	def __init__(self, cfg: PlateBallEnvCfg, render_mode: str | None = None, **kwargs):
		super().__init__(cfg, render_mode, **kwargs)
		self.first_reset = True

	def _setup_scene(self):
		self.plate = RigidObject(self.cfg.plate_cfg)
		self.ball = RigidObject(self.cfg.ball_cfg)

		spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

		# clone, filter, and replicate
		self.scene.clone_environments(copy_from_source=False)
		self.scene.filter_collisions(global_prim_paths=[])

		# add lights
		light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
		light_cfg.func("/World/Light", light_cfg)

	def _pre_physics_step(self, actions: torch.Tensor) -> None:
		self.new_state = self.plate.data.root_state_w.clone()
		curr_rot_angles = torch.concat([x.unsqueeze(-1) for x in euler_xyz_from_quat(self.new_state[:, 3:7])], dim=-1)
		new_angles = curr_rot_angles + actions * self.cfg.action_damping
		self.new_state[:, 3:7] = quat_from_euler_xyz(new_angles[:, 0], new_angles[:, 1], new_angles[:, 2])

		self.ball.update(self.cfg.dt)
		self.plate.update(self.cfg.dt)

	def _apply_action(self) -> None:
		self.plate.write_root_state_to_sim(self.new_state)

	def _get_observations(self) -> dict:
		ball_pos = self.ball.data.root_pos_w.clone()
		plate_rot = self.plate.data.root_quat_w.clone()
		plate_rot = torch.concat([x.unsqueeze(-1) for x in euler_xyz_from_quat(plate_rot)], dim=-1)
		return {"policy": torch.concat([ball_pos, plate_rot], dim=-1)}

	def _get_rewards(self) -> torch.Tensor:
		plate_pos = self.plate.data.root_pos_w.clone()
		ball_pos = self.ball.data.root_pos_w.clone()
		loss = F.mse_loss(plate_pos, ball_pos, reduction = "sum")
		return -loss

	def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
		out_of_bounds = self.ball.data.root_pos_w[:, -1] < self.cfg.fall_threshold
		time_out = self.episode_length_buf >= self.max_episode_length - 1
		return out_of_bounds, time_out

	def _reset_idx(self, env_ids: Sequence[int] | None):
		self.episode_length_buf[env_ids] = 0.0
		if self.first_reset:
			self.init_plate_position = self.plate.data.root_state_w.clone()
			self.init_ball_position = self.ball.data.root_state_w.clone()
			self.first_reset = False
			return
			
		rand_vel = torch.randn(len(env_ids), 6, device="cuda") * 0.2
		self.plate.write_root_state_to_sim(self.init_plate_position[env_ids], env_ids)
		self.ball.write_root_state_to_sim(self.init_ball_position[env_ids], env_ids)
		self.ball.write_root_com_velocity_to_sim(rand_vel, env_ids)

		