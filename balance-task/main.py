import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn.functional as F
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.scene import InteractiveScene
from scene import PlateBallSceneCfg

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
	sim_dt = sim.get_physics_dt()
	count = 0
	plate = scene["plate"]
	ball = scene["ball"]

	init_pos_plate = plate.data.root_state_w.clone()
	init_pos_ball = ball.data.root_state_w.clone()
	while simulation_app.is_running():
		if count % 500 == 0:
			count = 0

			plate.write_root_state_to_sim(init_pos_plate)
			ball.write_root_state_to_sim(init_pos_ball)

			print("[INFO]: Resetting robot state...")

		root_state = plate.data.root_state_w.clone()
		delta_angle = ((torch.rand(root_state.shape[0], 4, device="cuda") * 2) - 1.0) * 0.001
		new_quat = root_state[:, 3:7] + delta_angle
		new_quat = F.normalize(new_quat, p=2, dim=-1)
		root_state[:, 3:7] = new_quat

		plate.write_root_state_to_sim(root_state)

		scene.write_data_to_sim()
		sim.step()
		count += 1
		scene.update(sim_dt)


def main():
	sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
	sim = SimulationContext(sim_cfg)
	sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

	scene_cfg = PlateBallSceneCfg(num_envs=1024, env_spacing=2.0)
	scene = InteractiveScene(scene_cfg)

	sim.reset()
	print("[INFO]: Setup complete...")
	run_simulator(sim, scene)


if __name__ == "__main__":
	main()
	simulation_app.close()