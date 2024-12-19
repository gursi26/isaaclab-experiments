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
from omni.isaac.lab.envs import DirectRLEnv
from env import PlateBallEnvCfg, PlateBallEnv

def run_simulator(env):
	count = 0
	while simulation_app.is_running():
		rand_delta = torch.randn(env.num_envs, 3, device="cuda")
		obs, rew, reset_term, reset_trunc, extras = env.step(rand_delta)
		# print("-" * 50)
		# print(obs["policy"].shape)
		print(rew)
		# print(reset_term)
		# print(reset_trunc)
		# print(extras)
		# print("-" * 50)
		count += 1


def main():
	env_cfg = PlateBallEnvCfg()
	env = PlateBallEnv(env_cfg)
	env.reset()
	run_simulator(env)


if __name__ == "__main__":
	main()
	simulation_app.close()