import os
import gym
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # uncomment for a no-gpu option


# Select game
GAME_ASSAULT = "Assault-v0"
GAME = GAME_ASSAULT

if GAME == GAME_ASSAULT:
    # An observation is the image that is fed into the dqn.
    OBSERVATION_WIDTH = 210
    OBSERVATION_HEIGHT = 160
    OBSERVATION_CHANNELS = 3
    OBSERVATION_SHAPE = (OBSERVATION_WIDTH, OBSERVATION_HEIGHT, OBSERVATION_CHANNELS)


######################################################################################################
#                                           Pipeline
######################################################################################################
def ProcessObservation(observation):
    """
    Description:
        Takes in an observation of shape OBSERVATION_SHAPE and outputs a processed observation
    Returns:
        object  - The processed observation
    """

    # TODO: implementation
    return observation


######################################################################################################
#                                           Environment
######################################################################################################
def SetupEnvironment():
    """
    Description:
        Sets up the gym environment and renders it in a human understandable
        format (i.e. proper framerate)
    Returns:
        object  - The gym environment object
        object  - The first observation after reseting the environment
    """

    env = gym.make(GAME, render_mode='human')
    initialObservation = env.reset()
    return env, initialObservation


def RunEnvironment(env, agent, initialObservation, numSteps=100):
    """
    Description:
        Runs the gym environment
    Parameters:
        env (object)                     - The gym environment object
        agent (DuelingDQNAgent)          - The agent to use on the environment
        initialObservation (object)      - The first observation after reseting the environment
        numSteps (int)                   - The number of steps to run
    Returns:
        None
    """

    observation = initialObservation  # the current image

    for _ in range(numSteps):
        env.render() # render each frame in a window
        
        observation, reward, done, info = env.step(agent.Forward(observation))

        # TODO: we can use ProcessObservation to apply preprocessing on the observations (images) before we run them through our dqn if needed
        observation = ProcessObservation(observation)

        #TODO: update the agent here


######################################################################################################
#                                           Dueling DQN
######################################################################################################
class DuelingDQNAgent():
    def __init__(self, env, exploreRate=0.8, exploreDecay=0.995, exploreMin=0.01, learningRate=0.01, batchSize=20):
        """
        Description:
            Initialized a Dueling DQN Agent
        Parameters:
            env (object)                - The gym environment object
            exploreRate (float)         - The gym environment object
        """
        self.env = env  # the gym environment
        self.numActions = self.env.action_space.n

        self.exploreRate = exploreRate
        self.exploreDecay = exploreDecay
        self.exploreMin = exploreMin
        self.lr = learningRate
        self.batchSize = batchSize


        self.model_CNN, cnnOutShape = self.CreateModel_CNN()
        self.model_StateValue = self.CreateModel_StateValue(cnnOutShape)
        self.model_ActionAdvantage = self.CreateModel_ActionAdvantage(cnnOutShape, self.numActions)
    

    def CreateModel_CNN(self):
        """
        Description:
            The first layer that an observation is passed through for the dueling DQN. It takes in an observation and outputs a state
        Returns:
            object  - The neural network model
        """

        model = keras.Sequential()
        lossType = keras.losses.categorical_crossentropy
        print(f"IN SHAPE: {OBSERVATION_SHAPE}")
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="elu", input_shape=OBSERVATION_SHAPE))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="elu"))
        model.add(keras.layers.Flatten())
        model.compile(optimizer='adam', loss=lossType)
        return model, model.layers[-1].output_shape


    def CreateModel_StateValue(self, inputShape):
        """
        Description:
            Creates a fully connected neural network that outputs a scalar value for a given a state
        Parameters:
            inputShape (object)    - The output shape from the convolutional neural network that is used as an input for this fully connected layer
        Returns:
            object  - The neural network model
        """

        model = keras.Sequential()
        lossType = keras.losses.categorical_crossentropy
        model.add(keras.layers.Dense(128, activation="relu", input_shape=inputShape))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer='adam', loss=lossType)
        return model


    def CreateModel_ActionAdvantage(self, inputShape, numActions):
        """
        Description:
            Creates a fully connected neural network that outputs a probability for each action given a state
        Parameters:
            inputShape (object)    - The output shape from the convolutional neural network that is used as an input for this fully connected layer
            numActions (int)       - The number of actions in the action space. It controls the number of outputs from this layer.
        Returns:
            object  - The neural network model
        """

        model = keras.Sequential()
        lossType = keras.losses.categorical_crossentropy
        model.add(keras.layers.Dense(128, activation="relu", input_shape=inputShape))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(numActions, activation="softmax"))
        model.compile(optimizer='adam', loss=lossType)
        return model

    
    def Forward(self, observation):
        """
        Description:
            Calculates the next action to perform by running through all three neural networks.
            Will randomly select an action based on exploreRate
        Parameters:
            observation (object)    - The observation read from the gym environment of shape OBSERVATION_SHAPE
        Returns:
            int  - The action that should be performed on the environment
        """

        if np.random.rand() < self.exploreRate:
            return random.randrange(self.numActions)  # randomly select one of the actions

        # The variables below are not descriptive, but instead follow formula 8 (pg 4) from this paper  https://arxiv.org/pdf/1511.06581.pdf
        s = self.model_CNN.predict(observation.reshape((1, OBSERVATION_WIDTH, OBSERVATION_HEIGHT, OBSERVATION_CHANNELS)))  # reshape observation to be compatible with keras conv layer
        V = self.model_StateValue.predict(s)       # get the value associated with the state
        A = self.model_ActionAdvantage.predict(s)  # get all advantages associated with a state
        Q = V + (A - np.argmax(A))  # calculate Q-Values (Q is an array)
        return np.argmax(Q)  # return the action that has the highest Q-Value


    def Backward(self):
        # TODO: Implementation
        pass


######################################################################################################
#                                           Main
######################################################################################################
if __name__ == '__main__':
    env, initialObservation = SetupEnvironment()

    print(f"\nAction space size: {env.action_space.n}\n")

    duelingDQN = DuelingDQNAgent(env)

    RunEnvironment(env, duelingDQN, initialObservation)

    env.close()