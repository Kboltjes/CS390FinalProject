import os
import gym
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from keras import Sequential, Model
from keras.layers import Input, Dense, Add, Conv2D, Flatten

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # uncomment for a no-gpu option

MODEL_FILENAME = "TrainedModel"
DO_RUN_TEST = False

# Select game
GAME_ASSAULT = "Assault-v0"
GAME_SPACE_INVADERS = "SpaceInvaders-v0"
GAME = GAME_ASSAULT

if GAME == GAME_ASSAULT or GAME == GAME_SPACE_INVADERS:
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

    observation = observation.reshape((1, OBSERVATION_WIDTH, OBSERVATION_HEIGHT,
                                       OBSERVATION_CHANNELS))
    return observation


def ProcessObservations(observation):
    """
    Description:
        Takes in an observation of shape OBSERVATION_SHAPE and outputs a processed observation
    Returns:
        object  - The processed observation
    """

    observation = observation.reshape((-1, OBSERVATION_WIDTH, OBSERVATION_HEIGHT,
                                       OBSERVATION_CHANNELS))
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

    env = gym.make(GAME)#, render_mode='human')

    initialObservation = env.reset()
    return env, initialObservation


######################################################################################################
#                                           Dueling DQN
######################################################################################################
class Memory:
    def __init__(self, capacity=2 ** 14):
        self.samples = []
        self.capacity = capacity

    def Remember(self, obs, action, reward, nextObs, done):
        """
        Adds elements to list. If list has reached capacity, we discard the first observations present
        This can be improved later by using an np.ndarray instead of a list
        """
        self.samples.append((obs, action, reward, nextObs, done))
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def Sample(self, size):
        """
        Args:
            size: Should be equivalent to batch size that'll be used in training

        Returns
            list - `size` number of random observations from the list
            None - if `size` is larger than the current size of the list
        """
        if size > len(self.samples):
            return None

        sample = random.sample(self.samples, size)
        sampleList = zip(*sample)
        return map(np.asarray, sampleList)

    def __len__(self):
        return len(self.samples)


######################################################################################################
#                                           Dueling DQN
######################################################################################################
class DuelingDQN:
    def __init__(self, numActions, exploreRate, exploreMin, exploreDecay, doTest=False):
        """
        Description:
            Initialized a Dueling DQN Model
        Parameters:
            numActions (int)         - The number of actions the environment has
        """

        self.numActions = numActions
        self.exploreRate = exploreRate
        self.exploreMin = exploreMin
        self.exploreDecay = exploreDecay

        if doTest:
            self.model = None # this will be loaded in the Load() function
        else:
            self.model = self.CreateModel()

    def CreateModel(self):
        input = Input(OBSERVATION_SHAPE)
        lossType = keras.losses.categorical_crossentropy

        backbone_1 = Conv2D(16, kernel_size=(3, 3), activation="elu")(input)
        backbone_2 = Conv2D(16, kernel_size=(3, 3), activation="elu")(backbone_1)
        backbone_3 = Flatten()(backbone_2)

        # Backbone is now fed into two separate 'networks'
        value_1 = Dense(64, activation="relu")(backbone_3)
        value_2 = Dense(32, activation="relu")(value_1)
        value_3 = Dense(1, activation="sigmoid")(value_2)

        adv_1 = Dense(64, activation="relu")(backbone_3)
        adv_2 = Dense(32, activation="relu")(adv_1)
        adv_3 = Dense(self.numActions, activation="softmax")(adv_2)

        q_layer_3 = tf.add(value_3, tf.subtract(adv_3, tf.reduce_mean(adv_3)))

        model = Model(inputs=input, outputs=q_layer_3)
        model.summary()
        model.compile(optimizer='adam', loss=lossType)
        return model

    def MakeMove(self, observation):
        """
        Description:
            Calculates the next action to perform by running through all three neural networks.
            Will randomly select an action based on exploreRate
        Parameters:
            observation (object)    - The observation read from the gym environment of shape OBSERVATION_SHAPE
        Returns:
            int  - The action that should be performed on the environment
        """
        self.exploreRate = max(self.exploreMin, self.exploreRate * self.exploreDecay)
        if np.random.rand() < self.exploreRate:
            return random.randrange(self.numActions)  # randomly select one of the actions
        Q = self.model.predict(observation)
        #print(Q)
        return np.argmax(Q)

    def Predict(self, observation):
        return self.model.predict(observation)

    def Train(self, x, y):
        self.model.fit(x, y, epochs=1, verbose=0)

    def Save(self):
        self.model.save(MODEL_FILENAME)

    def LoadModel(self):
        self.model = keras.models.load_model(MODEL_FILENAME)


class DuelingDQNAgent:
    def __init__(self, env, exploreRate=1.0, exploreDecay=0.995, exploreMin=0.01, batchSize=20, updateTime=20, doTest=False):
        """
        Description:
            Initialized a Dueling DQN Agent
        Parameters:
            env (object)                - The openai gym object
            exploreRate (float)         - How frequently on a scale [0-1] that it selects a random action to explore
            exploreDecay (float)        - How fast the exploreRate decays
            exploreMin (float)          - The lowest value that exploreRate can decay down to
            batchSize (int)             - The batch size to use for learning
            doTest (bool)               - If the agent should load a saved model and only test it without training
        """

        self.env = env
        self.memory = Memory()

        self.exploreRate = exploreRate
        self.exploreDecay = exploreDecay
        self.exploreMin = exploreMin
        self.batchSize = batchSize

        self.numActions = self.env.action_space.n
        self.updateTime = updateTime

        if doTest: # load a saved model
            self.model = DuelingDQN(self.numActions, 0, 0, 0, doTest=True)
            self.model.LoadModel()
        else: # create a new model
            self.model = DuelingDQN(self.numActions, exploreRate, exploreMin, exploreDecay)
            self.targetModel = DuelingDQN(self.numActions, exploreRate, exploreMin, exploreDecay)
            self.Update()

    def Update(self):
        self.targetModel.model.set_weights(self.model.model.get_weights())

    def MakeMove(self, observation):
        self.targetModel.MakeMove(observation)

    def PlayGame(self):
        self.env.reset()

    def Test(self, initialObservation):
        observation = ProcessObservation(initialObservation)  # the current image
        done = False
        while not done:
            env.render()  # render each frame in a window
            action = self.model.MakeMove(observation)
            print(action)
            observation, _, done, _ = env.step(action)

            observation = ProcessObservation(observation)

        print("Finished Testing!")


    def Forward(self, initialObservation, numSteps=30000):
        """
        Description:
            Runs the gym environment
        Parameters:
            initialObservation (object)      - The first observation after resetting the environment
            numSteps (int)                   - The number of steps to run
        Returns:
            None
        """

        observation = ProcessObservation(initialObservation)  # the current image
        count = 0
        for _ in range(numSteps):
            env.render()  # render each frame in a window

            prevObservation = observation
            action = self.model.MakeMove(observation)
            observation, reward, done, _ = env.step(action)
            #print(f"Reward: {reward}, Action: {action}")

            if done:
                observation = env.reset()

            observation = ProcessObservation(observation)

            self.memory.Remember(prevObservation, action, reward, observation, done)  # Add to memory

            count += 1
            if count == self.updateTime:
                self.Backward()
                self.Update()
                count = 0

    def Backward(self):
        """
        Description:
            Trains the model on previous observations
        Returns:
            None
        """
        if len(self.memory) < self.batchSize:
            return

        observs, actions, rewards, nextObservs, dones = self.memory.Sample(self.batchSize)
        observs = ProcessObservations(observs)
        nextObservs = ProcessObservations(nextObservs)
        targets = self.targetModel.Predict(observs)
        nextTargets = self.targetModel.MakeMove(nextObservs)
        targets[range(self.batchSize), actions] = rewards + (1 - dones) * nextTargets * 0.95
        # FIXME: Look at this formula once again
        self.model.Train(observs, targets)

    def SaveModel(self):
        self.model.Save()


######################################################################################################
#                                           Main
######################################################################################################
if __name__ == '__main__':
    startTime = datetime.now()
    env, initialObservation = SetupEnvironment()

    print(f"\nObservation Shape: {initialObservation.shape}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Actions: {env.unwrapped.get_action_meanings()}\n")

    if DO_RUN_TEST:
        agent = DuelingDQNAgent(env, doTest=True)
        agent.Test(initialObservation)
    else: # train the model then save it
        agent = DuelingDQNAgent(env, doTest=False)
        agent.Forward(initialObservation)
        agent.SaveModel()

    env.close()

    print(f"Duration: {datetime.now() - startTime}")
