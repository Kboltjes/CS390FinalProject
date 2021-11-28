import os
import gym
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense, Add, Subtract, Maximum, Conv2D, Flatten

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

    observation = ProcessObservation(initialObservation)  # the current image

    for _ in range(numSteps):
        env.render()  # render each frame in a window

        prevObservation = observation
        action = agent.Forward(observation)
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()

        # TODO: we can use ProcessObservation to apply preprocessing on the observations (images) before we run them through our dqn if needed
        observation = ProcessObservation(observation)

        agent.memory.remember(prevObservation, action, reward, observation, done)  # Add to memory
        agent.Backward()


######################################################################################################
#                                           Dueling DQN
######################################################################################################
class Memory:
    def __init__(self, capacity=2 ** 14):
        self.samples = []
        self.capacity = capacity

    def remember(self, obs, action, reward, nextObs, done):
        """
        Adds elements to list. If list has reached capacity, we discard the first observations present
        This can be improved later by using an np.ndarray instead of a list
        """
        self.samples.append((obs, action, reward, nextObs, done))
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, size):
        """
        Args:
            size: Should be equivalent to batch size that'll be used in training

        Returns
            list - `size` number of random observations from the list
            None - if `size` is larger than the current size of the list
        """
        if size > len(self.samples):
            return None

        return random.sample(self.samples, size)

    def __len__(self):
        return len(self.samples)


######################################################################################################
#                                           Dueling DQN
######################################################################################################
class DuelingDQNAgent:
    def __init__(self, env, exploreRate=0.8, exploreDecay=0.995, exploreMin=0.01, batchSize=20):
        """
        Description:
            Initialized a Dueling DQN Agent
        Parameters:
            env (object)                - The gym environment object
            exploreRate (float)         - How frequently on a scale [0-1] that it selects a random action to explore
            exploreDecay (float)        - How fast the exploreRate decays
            exploreMin (float)          - The lowest value that exploreRate can decay down to
            batchSize (int)             - The batch size to use for learning
        """

        self.env = env  # the gym environment
        self.memory = Memory()
        self.numActions = self.env.action_space.n

        self.exploreRate = exploreRate
        self.exploreDecay = exploreDecay
        self.exploreMin = exploreMin
        self.batchSize = batchSize

        self.model = self.CreateModel(self.numActions)
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

    def CreateModel(self, numActions):
        input = Input(OBSERVATION_SHAPE)
        lossType = keras.losses.categorical_crossentropy

        backbone_1 = Conv2D(32, kernel_size=(3, 3), activation="elu")(input)
        backbone_2 = Conv2D(64, kernel_size=(3, 3), activation="elu")(backbone_1)
        backbone_3 = Flatten()(backbone_2)

        value_1 = Dense(128, activation="relu")(backbone_3)
        value_2 = Dense(64, activation="relu")(value_1)
        value_3 = Dense(1, activation="sigmoid")(value_2)

        adv_1 = Dense(128, activation="relu")(backbone_3)
        adv_2 = Dense(64, activation="relu")(adv_1)
        adv_3 = Dense(numActions, activation="softmax")(adv_2)

        # q_layer_1 = Maximum()([adv_3])
        # q_layer_2 = Subtract()[adv_3, q_layer_1]
        q_layer_3 = Add()([value_3, adv_3])
        # q_layer = Maximum(q_layer_3)

        model = keras.Model(inputs=input, outputs=q_layer_3)
        model.summary()
        model.compile(loss=lossType, optimzer=keras.optimizers.Adam())
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

        # # The variables below are not descriptive, but instead follow formula 8 (pg 4) from this paper  https://arxiv.org/pdf/1511.06581.pdf
        # s = self.model_CNN.predict(observation.reshape((1, OBSERVATION_WIDTH, OBSERVATION_HEIGHT,
        #                                                 OBSERVATION_CHANNELS)))  # reshape observation to be compatible with keras conv layer
        # V = self.model_StateValue.predict(s)  # get the value associated with the state
        # A = self.model_ActionAdvantage.predict(s)  # get all advantages associated with a state
        # Q = V + (A - np.argmax(A))  # calculate Q-Values (Q is an array)
        # return np.argmax(Q)  # return the action that has the highest Q-Value
        Q = self.model.predict(observation.reshape((1, OBSERVATION_WIDTH, OBSERVATION_HEIGHT, OBSERVATION_CHANNELS)))
        return np.argmax(Q)

    def Backward(self):

        if len(self.memory) > self.batchSize:
            return

        # Model training
        samples = self.memory.sample(self.batchSize)
        loss = self._loss(samples)

        # Update exploreRate
        self.exploreRate = max(self.exploreMin, self.exploreRate * self.exploreDecay)

    def _loss(self, samples):

        return self


######################################################################################################
#                                           Main
######################################################################################################
if __name__ == '__main__':
    env, initialObservation = SetupEnvironment()

    print(f"\nAction space size: {env.action_space.n}\n")

    duelingDQN = DuelingDQNAgent(env)

    RunEnvironment(env, duelingDQN, initialObservation)

    env.close()
