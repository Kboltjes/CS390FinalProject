import os
import gym
import numpy as np
import pandas as pd

# An observation is the image that is fed into the dqn.
OBSERVATION_WIDTH = 210
OBSERVATION_HEIGHT = 160
OBSERVATION_CHANNELS = 3
OBSERVATION_SHAPE = (OBSERVATION_WIDTH, OBSERVATION_HEIGHT, OBSERVATION_CHANNELS)


######################################################################################################
#                                           Pipeline
######################################################################################################
def ProcessObservation(observation):
    '''
    Description:
        Takes in an observation of shape OBSERVATION_SHAPE and outputs a processed observation
    Returns:
        The processed observation
    '''

    # TODO: implementation
    return observation


######################################################################################################
#                                           Environment
######################################################################################################
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
        
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action

        # TODO: we can use ProcessObservation to apply preprocessing on the observations (images) before we run them through our dqn
        processedObservation = ProcessObservation(observation)

        if done:
            print("Finished Running Algorithm")
            break


######################################################################################################
#                                           Main
######################################################################################################
if __name__ == '__main__':
    env = SetupEnvironment()

    # just do a random test to make sure the library is working properly
    RandomSampleEnvTest(env)

    env.close()