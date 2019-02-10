"""
An intro file to explore OpenAI gym environments
"""

import numpy as np
import matplotlib.pyplot as plt
import gym

available_envs = gym.envs.registry.all()
env_ids = [env_spec.id for env_spec in available_envs]

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print_env_info(env):
    print("Action space: ", env.action_space)
    print("Observation space: ", env.observation_space)
    print("Reward range: ", env.reward_range)

def main():
    env = gym.make('CartPole-v0')
    print_env_info(env)
    initial_obs = env.reset()
    print("Initial observation", initial_obs)

if __name__ == '__main__':
    print("executing ", __file__)
    main()
