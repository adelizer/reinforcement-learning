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
    tmax = 1000
    print_env_info(env)
    random_policies = np.random.uniform(-1,1,(500,env.observation_space.shape[0]))
    total_reward = []
    for i in range(random_policies.shape[0]):
        obs = env.reset()
        done = False
        episode_reward = 0
        while tmax>0 or not done:
            action = int(np.dot(obs, random_policies[i,:])>0)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            tmax-=1
        total_reward.append(episode_reward)
    idx = np.argmax(total_reward)
    print("Max reward values: {} \n With policy number {} \n "
          "The following parameters: {}".format(total_reward[idx],
                                                idx,
                                                random_policies[idx,:]))
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(int(np.dot(random_policies[idx,:], obs)>0))
    env.env.close()

if __name__ == '__main__':
    print("executing ", __file__)
    main()
