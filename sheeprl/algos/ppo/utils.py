from typing import List

import gymnasium as gym
import torch
from lightning import Fabric

from sheeprl.algos.ppo.agent import PPOAgent
from sheeprl.algos.ppo.args import PPOArgs


@torch.no_grad()
def test(agent: PPOAgent, env: gym.Env, fabric: Fabric, args: PPOArgs, cnn_keys: List[str], mlp_keys: List[str]):
    agent.eval()
    done = False
    cumulative_rew = 0
    o = env.reset(seed=args.seed)[0]
    obs = {}
    for k in o.keys():
        if k in mlp_keys + cnn_keys:
            with fabric.device:
                torch_obs = torch.from_numpy(o[k]).view(1, *o[k].shape)
            obs[k] = torch_obs / 255 - 0.5 if k in cnn_keys else torch_obs.float()
    while not done:
        # Act greedly through the environment
        actions = torch.cat([act.argmax(dim=-1) for act in agent.get_greedy_actions(obs)], dim=-1)

        # Single environment step
        o, reward, done, truncated, _ = env.step(actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        obs = {}
        for k in o.keys():
            if k in mlp_keys + cnn_keys:
                with fabric.device:
                    torch_obs = torch.from_numpy(o[k]).view(1, *o[k].shape).float()
                obs[k] = torch_obs / 255 - 0.5 if k in cnn_keys else torch_obs

        if args.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()
