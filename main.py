import os
import gym
import numpy as np
import pandas as pd


def SetupEnvironment():
    '''
    Description:
        Sets up the gym environment and renders it in a human understandable
        format (i.e. proper framerate)
    Returns:
        The gym environment object
    '''
    env = gym.make('Assault-v0', render_mode='human')
    env.reset()
    return env


def RandomSampleEnvTest(env):
    '''
    Description:
        Performs a test of the gym environment by stepping through it by taking random 
        actions from the action space.
    Parameters:
        env (object)    - The gym environment object
    Returns:
        None
    '''
    for _ in range(1000):
        env.render() # render each frame in a window
        env.step(env.action_space.sample()) # take a random action


if __name__ == '__main__':
    env = SetupEnvironment()

    # just do a random test to make sure the library is working properly
    RandomSampleEnvTest(env)

    env.close()