from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections.abc import Sequence


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat

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
			collision_props=sim_utils.CollisionPropertiesCfg(
				collision_enabled=True
			),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
			activate_contact_sensors=True
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
			collision_props=sim_utils.CollisionPropertiesCfg(
				collision_enabled=True
			),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
			activate_contact_sensors=True
		),
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(0, 0, 3)
		),
	)
	return cfg


@configclass
class PlateBallEnvCfg(DirectRLEnvCfg):
	num_envs = 1024
	env_spacing = 4.0
	dt = 1 / 120
	observation_space = 15
	action_space = 3
	state_space = 0

	# env
	decimation = 1
	episode_length_s = 15.0

	# simulation
	sim: SimulationCfg = SimulationCfg(
		dt=dt, 
		render_interval=decimation
	)

	plate_cfg: RigidObjectCfg = _get_plate_config()
	ball_cfg: RigidObjectCfg = _get_ball_config()
	contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Plate",
        update_period=0.0005,
		history_length=1,
		track_air_time=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Ball"],
    )
	fall_threshold = 1.5
	action_damping = 0.02

	# reward weights
	alive_weight = 1.0
	dead_weight = -10.0
	pos_err_weight = -3.0
	vel_weight = -1.0
	ang_vel_weight = -1.0
	movement_weight = -1.0

	# scene
	scene: InteractiveSceneCfg = InteractiveSceneCfg(
		num_envs=num_envs, 
		env_spacing=env_spacing, 
		replicate_physics=True,
	)

# get current angle orientation of object
def get_angles(quat):
	angles_ = euler_xyz_from_quat(quat)
	angles = torch.cat([a.unsqueeze(-1) for a in angles_], dim=-1)
	return angles

# projects the balls 3d location to a 2d point on the plate
def project_to_plane(
    plane_center: torch.Tensor,  # [N, 3]
    plane_angles: torch.Tensor,  # [N, 3], roll, pitch, yaw
    ball_center: torch.Tensor    # [N, 3]
):
    # Extract pitch, roll, yaw angles
    roll, pitch, yaw = plane_angles[:, 0], plane_angles[:, 1], plane_angles[:, 2]
    
    # Compute rotation matrices for each axis
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    R_pitch = torch.stack([
        torch.stack([torch.ones_like(cos_pitch), torch.zeros_like(cos_pitch), torch.zeros_like(cos_pitch)], dim=-1),
        torch.stack([torch.zeros_like(cos_pitch), cos_pitch, -sin_pitch], dim=-1),
        torch.stack([torch.zeros_like(cos_pitch), sin_pitch, cos_pitch], dim=-1)
    ], dim=1)

    R_roll = torch.stack([
        torch.stack([cos_roll, torch.zeros_like(cos_roll), sin_roll], dim=-1),
        torch.stack([torch.zeros_like(cos_roll), torch.ones_like(cos_roll), torch.zeros_like(cos_roll)], dim=-1),
        torch.stack([-sin_roll, torch.zeros_like(cos_roll), cos_roll], dim=-1)
    ], dim=1)

    R_yaw = torch.stack([
        torch.stack([cos_yaw, -sin_yaw, torch.zeros_like(cos_yaw)], dim=-1),
        torch.stack([sin_yaw, cos_yaw, torch.zeros_like(cos_yaw)], dim=-1),
        torch.stack([torch.zeros_like(cos_yaw), torch.zeros_like(cos_yaw), torch.ones_like(cos_yaw)], dim=-1)
    ], dim=1)

    R = torch.bmm(R_yaw, torch.bmm(R_pitch, R_roll))
    rel_pos = ball_center - plane_center
    local_pos = torch.bmm(R.transpose(1, 2), rel_pos.unsqueeze(-1)).squeeze(-1)
    projected_point = local_pos[:, :2]
    return projected_point

class PlateBallEnv(DirectRLEnv):
	cfg: PlateBallEnvCfg

	def __init__(self, cfg: PlateBallEnvCfg, render_mode: str | None = None, **kwargs):
		super().__init__(cfg, render_mode, **kwargs)
		self.first_reset = True
		self.predicted_actions = torch.zeros(self.num_envs, 3, device="cuda")

	def _setup_scene(self):
		self.plate = RigidObject(self.cfg.plate_cfg)
		self.ball = RigidObject(self.cfg.ball_cfg)
		self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
		self.scene.sensors["contact_sensor"] = self.contact_sensor

		spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

		# clone, filter, and replicate
		self.scene.clone_environments(copy_from_source=False)
		self.scene.filter_collisions(global_prim_paths=[])

		# add lights
		light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
		light_cfg.func("/World/Light", light_cfg)

	def _pre_physics_step(self, actions: torch.Tensor) -> None:
		self.predicted_actions = actions
		self.new_state = self.plate.data.root_state_w.clone()
		curr_rot_angles = get_angles(self.new_state[:, 3:7])
		new_angles = curr_rot_angles + actions * self.cfg.action_damping
		self.new_state[:, 3:7] = quat_from_euler_xyz(new_angles[:, 0], new_angles[:, 1], new_angles[:, 2])

	def _apply_action(self) -> None:
		self.plate.write_root_state_to_sim(self.new_state)
		self.ball.update(self.cfg.dt)
		self.plate.update(self.cfg.dt)

	def _get_observations(self) -> dict:
		ball_pos = get_angles(self.ball.data.root_state_w[:, 3:7].clone())
		ball_vel = self.ball.data.root_com_vel_w.clone()
		plate_rot = get_angles(self.plate.data.root_quat_w.clone())
		prev_action = self.predicted_actions.clone()
		return {"policy": torch.concat([ball_pos, ball_vel, plate_rot, prev_action], dim=-1)}

	def _get_rewards(self) -> torch.Tensor:
		not_in_contact = self.contact_sensor.data.net_forces_w.clone().reshape((self.num_envs, 3)).sum(dim=-1) == 0
		ball_pos = self.ball.data.root_pos_w.clone()
		ball_vel = self.ball.data.root_com_vel_w.clone()
		pos_vel, angular_vel = ball_vel[:, :3], ball_vel[:, 3:]

		plate_pos = self.plate.data.root_pos_w.clone()
		plate_angles = get_angles(self.plate.data.root_state_w[:, 3:7].clone())
		projected_ball_pos = project_to_plane(plate_pos, plate_angles, ball_pos) 

		alive_rew = 1.0 - not_in_contact.float()
		dead_rew = not_in_contact.float()
		pos_rew = (projected_ball_pos ** 2).sum(dim=-1)
		vel_rew = (pos_vel ** 2).sum(dim=-1)
		ang_vel_rew = (angular_vel ** 2).sum(dim=-1)
		movement_rew = (self.predicted_actions ** 2).sum(dim=-1)

		rew = alive_rew * self.cfg.alive_weight \
			+ dead_rew * self.cfg.dead_weight \
			+ pos_rew * self.cfg.pos_err_weight \
			+ vel_rew * self.cfg.vel_weight \
			+ ang_vel_rew * self.cfg.ang_vel_weight \
			+ movement_rew * self.cfg.movement_weight
		return rew

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