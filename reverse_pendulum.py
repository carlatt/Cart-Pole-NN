import time

import gym
import numpy as np
import string
import random

#from deap import base, creator, tools



if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, m = env.step(action)  # take a random action
    env.close()