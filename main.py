import gym
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from keras import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # uncomment for a no-gpu option

MODEL_FILENAME = "TrainedModel.h5"
DO_RUN_TEST = False

# Select game
GAME_ASSAULT = "Assault-v0"
GAME_SPACE_INVADERS = "SpaceInvaders-v0"
GAME_BREAKOUT = "Breakout-v0"
GAME = GAME_BREAKOUT

if GAME == GAME_ASSAULT or GAME == GAME_SPACE_INVADERS or GAME == GAME_BREAKOUT:
    # An observation is the image that is fed into the dqn.
    OBSERVATION_WIDTH = 210
    OBSERVATION_HEIGHT = 160
    OBSERVATION_CHANNELS = 3
    OBSERVATION_SHAPE = (OBSERVATION_WIDTH, OBSERVATION_HEIGHT, OBSERVATION_CHANNELS)

    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    RESIZED_CHANNELS = 1
    RESIZED_SHAPE = (RESIZED_WIDTH, RESIZED_HEIGHT, RESIZED_CHANNELS)

    MAX_GAME_FRAMES = 20000 # the maximum number of frames that a single game is allowed to run for before being auto-reset
    NUM_FRAMES_PER_PASS = 4 # the number of frames used in 
    INPUT_SHAPE = (RESIZED_WIDTH, RESIZED_HEIGHT, NUM_FRAMES_PER_PASS)


######################################################################################################
#                                           Gym Environment Wrapper
######################################################################################################
class GymEnvWrapper:
    def __init__(self):
        """
        Description:
            Creates an environment wrapper for the gym environment
        """
        self.env = gym.make('BreakoutDeterministic-v4')

        self.state = None  # the last 3 frames and the current frame of the environment all stacked together
        self.currentLives = 0

    def reset(self):
        """
        Description:
            Resets the gym environment for a new game.
        """

        self.currentLives = 0
        self.frame = self.env.reset()

        # since we need 4 frames to feed through the network, just duplicate the first one 4 times
        self.state = np.repeat(ProcessObservation(self.frame), NUM_FRAMES_PER_PASS, axis=2)
    
    def render(self):
        self.env.render()

    def step(self, action, renderMode='human'):
        """
        Description:
            Passes the action to the gym environment and saves the result to self.state.
            Will render the frame if renderMode is 'human'
        Arguments:
            action (int)            - The action to run
            renderMode (object)     - 'human' will render a screen, anything else won't render anything
        Returns:
            object      - The processed new frame as a result of that action
            int         - The unscaled reward from the action
            bool        - If the game has ended or not
            bool        - If the action caused a life to be lost
        """
        observation, reward, done, info = self.env.step(action)

        didLoseLife = True if info['lives'] < self.currentLives else done # didLoseLife is essentially the same as done. They both mean to recall the ball in
        self.currentLives = info['lives']

        observation = ProcessObservation(observation)
        self.state = np.append(self.state[:, :, 1:], observation, axis=2)

        if renderMode == 'human':
            self.env.render()

        return observation, reward, done, didLoseLife
    
    def close(self):
        """
        Description:
            Closes the environment
        """
        self.env.close()


######################################################################################################
#                                           Pipeline
######################################################################################################
def ProcessObservation(observation):
    """
    Description:
        Takes in an observation of shape OBSERVATION_SHAPE and outputs a processed observation
    Parameters:
        observation (object)  - A (210, 160, 3) observation from the gym environment
    Returns:
        object  - An (84, 84, 1) grayscale version of the observation
    """
    
    observation = observation.astype(np.uint8)
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # grayscale the image
    observation = observation[34:34+160, :160]  # crop image to (84, 84)
    observation = cv2.resize(observation, (RESIZED_WIDTH, RESIZED_HEIGHT), interpolation=cv2.INTER_NEAREST)
    observation = observation.reshape((*(RESIZED_WIDTH, RESIZED_HEIGHT), 1))
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
        object  - The gym environment object wrapper
    """

    env = GymEnvWrapper()
    env.reset()
    return env


######################################################################################################
#                                           Memory
######################################################################################################
class Memory:
    def __init__(self, capacity=2 ** 14):
        self.samples = []
        self.capacity = capacity
        self.numItemsPerSample = 5

    def Remember(self, obs, action, reward, done):
        """
        Description:
            Adds elements to list. If list has reached capacity, we discard the first observations present
            This can be improved later by using an np.ndarray instead of a list
        """

        # scale the reward to be -1, 0, or 1
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        self.samples.append((obs, action, reward, done))
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def Sample(self, size):
        """
        Description:
            Returns a random sample from the memory.
        Parameters:
            size: Should be equivalent to batch size that'll be used in training
        Returns:
            list - `size` number of random observations from the list
            None - if `size` is larger than the current size of the list
        """
        if size > len(self.samples):
            return None

        sampleList = []
        for _ in range(size):
            index = random.randint(NUM_FRAMES_PER_PASS, len(self.samples) - 2) 
            l = list(self.samples[index]) # convert to a list, so we can append to the tuple
            
            # convert the observation from the sample into the 3 sequential observations leading up to it and itself
            # also append nextObservation while doing the same thing
            observations = []
            nextObservations = []
            for i in reversed(range(NUM_FRAMES_PER_PASS)):
                observations.append(self.samples[index - i][0])
                nextObservations.append(self.samples[index - i + 1][0]) # nextObservation is just 1 observation ahead of observations
            observations = np.asarray(observations)
            nextObservations = np.asarray(nextObservations)
            l[0] = observations
            l.append(nextObservations)
            sampleList.append(tuple(l)) # convert to tuple and append

        sampleArray = np.asarray(sampleList)

        # return the columns of sampleArray
        retVals = []
        for i in range(self.numItemsPerSample):
            retVals.append(sampleArray[:,i])

        # convert the observation tuples shape of (4, 84, 84) to (84, 84, 4)
        for i in range(size):
            retVals[0][i] = retVals[0][i].transpose(1, 2, 0)
            retVals[4][i] = retVals[4][i].transpose(1, 2, 0)

        return np.stack(retVals[0]), retVals[1], retVals[2], retVals[3], np.stack(retVals[4])

    def __len__(self):
        return len(self.samples)


######################################################################################################
#                                           Dueling DQN
######################################################################################################
class DuelingDQN:
    def __init__(self, numActions, exploreRate, exploreMin, exploreDecay, inputShape, doTest=False):
        """
        Description:
            Initialized a Dueling DQN Model
        Parameters:
            numActions (int)            - The number of actions the environment has
            exploreRate (float)         - How frequently on a scale [0-1] that it selects a random action to explore
            exploreMin (float)          - The lowest value that exploreRate can decay down to
            exploreDecay (float)        - How fast the exploreRate decays - NOT used with the new linear decay rate
            inputShape (object)         - The shape of input that the model will receive
            doTest (bool)               - If the agent should load a saved model and only test it without training
        """

        self.numActions = numActions
        self.exploreRate = exploreRate
        self.exploreMin = exploreMin
        self.exploreDecay = exploreDecay
        self.inputShape = inputShape

        if doTest:
            self.model = None # this will be loaded in the Load() function
        else:
            self.model = self.CreateModel()


    def CreateModel(self):
        """
        Description:
            Creates a Keras Dueling DQN model
        Returns:
            object  - The Keras Dueling DQN model
        """
        input = Input(shape=INPUT_SHAPE)


        normalizedInput = Lambda(lambda layer: layer / 255)(input)  # normalize everything to be in range [0-1]

        backbone_1 = Conv2D(32, kernel_size=(3, 3), activation="elu")(normalizedInput)
        backbone_2 = Conv2D(32, kernel_size=(3, 3), activation="elu")(backbone_1)
        backbone_3 = Flatten()(backbone_2)

        # Backbone is now fed into two separate 'networks'
        value_1 = Dense(64, activation="relu")(backbone_3)
        value_2 = Dense(32, activation="relu")(value_1)
        value_3 = Dense(1, activation="sigmoid")(value_2)

        adv_1 = Dense(64, activation="relu")(backbone_3)
        adv_2 = Dense(32, activation="relu")(adv_1)
        adv_3 = Dense(self.numActions, activation="softmax")(adv_2)

        q_layer_3 = tf.add(value_3, tf.subtract(adv_3, tf.reduce_mean(adv_3, axis=1, keepdims=True)))

        model = Model(inputs=input, outputs=q_layer_3)
        model.summary()
        model.compile(Adam(0.00001), loss=tf.keras.losses.Huber())
        return model

    def MakeMove(self, observation, frameNum, doUpdateExploreRate=True):
        """
        Description:
            Calculates the next action to perform by running through all three neural networks.
            Will randomly select an action based on exploreRate.
            Decreases explore rate if doUpdateExploreRate is true.
        Parameters:
            observation (object)        - The observation read from the gym environment of shape OBSERVATION_SHAPE
            frameNum (int)              - The frame number of training that the model is on (used for linear decay of explore rate)
            doUpdateExploreRate (bool)  - Whether explore rate should be decayed or not
        Returns:
            int  - The action that should be performed on the environment
        """

        if doUpdateExploreRate:
            self.UpdateExploreRate(frameNum)

        if np.random.rand() < self.exploreRate:
            return random.randrange(self.numActions)  # randomly select one of the actions

        Q = self.model.predict(observation.reshape((-1, RESIZED_WIDTH, RESIZED_HEIGHT, NUM_FRAMES_PER_PASS)))[0]
        return np.argmax(Q)

    def MakeMoveTest(self, observation):
        """
        Description:
            Calculates the next action to perform by running through all three neural networks.
            Does not do any random actions because this is meant for testing.
        Parameters:
            observation (object)    - The observation read from the gym environment of shape OBSERVATION_SHAPE
        Returns:
            int  - The action that should be performed on the environment
        """

        Q = self.model.predict(observation.reshape((-1, RESIZED_WIDTH, RESIZED_HEIGHT, NUM_FRAMES_PER_PASS)))[0]
        return np.argmax(Q)

    def UpdateExploreRate(self, frameNum):
        """
        Description:
            Updates the exploration rate linearly
        """
        # using a linearly decaying explore rate seems better than exponential
        # this gets us to an exploreRate of 0.1 at 1,000,000 frames
        self.exploreRate = -(0.0000009) * frameNum + 1  

    def Predict(self, observation):
        """
        Description:
            Provides predictions from the model
        """
        return self.model.predict(observation)

    def Save(self):
        """
        Description:
            Saves the model to a file
        """
        self.model.save(MODEL_FILENAME)

    def LoadModel(self):
        """
        Description:
            Loads a model from a file rather than creating a new one
        """
        self.model = keras.models.load_model(MODEL_FILENAME)


class DuelingDQNAgent:
    def __init__(self, env, exploreRate=1.0, exploreDecay=0.999999, exploreMin=0.1, batchSize=32, trainTime=4, updateTime=5000, startBatchSize=50000, doTest=False):
        """
        Description:
            Initialized a Dueling DQN Agent
        Parameters:
            env (object)                - The openai gym object
            exploreRate (float)         - How frequently on a scale [0-1] that it selects a random action to explore
            exploreDecay (float)        - How fast the exploreRate decays - NOT used with the new linear decay rate
            exploreMin (float)          - The lowest value that exploreRate can decay down to
            batchSize (int)             - The batch size to use for learning
            trainTime (int)             - The number of frames to run in between training model
            updateTime (int)            - The number of frames to run in between updating the target network
            startBatchSize (int)        - The number of random frames to store into memory before training begins
            doTest (bool)               - If the agent should load a saved model and only test it without training
        """

        self.env = env

        self.numActions = self.env.env.action_space.n
        self.memory = Memory()

        self.exploreRate = exploreRate
        self.exploreDecay = exploreDecay
        self.exploreMin = exploreMin
        self.batchSize = batchSize
        self.startBatchSize = startBatchSize

        self.updateTime = updateTime

        if doTest: # load a saved model
            self.model = DuelingDQN(self.numActions, 0, 0, 0, self.env.env.observation_space.shape, doTest=True)
            self.model.LoadModel()
        else: # create a new model
            self.model = DuelingDQN(self.numActions, exploreRate, exploreMin, exploreDecay, self.env.env.observation_space.shape)
            self.targetModel = DuelingDQN(self.numActions, exploreRate, exploreMin, exploreDecay, self.env.env.observation_space.shape)
            self.Update()

    def Update(self):
        """
        Description:
            Updates the target model weights
        """
        self.targetModel.model.set_weights(self.model.model.get_weights())

    def Test(self):
        """
        Description:
            Runs a test on a model in a single game
        """
        env.reset()
        done = False
        didLoseLife = False
        while not done:
            if didLoseLife:
                # force the environment to resummon the ball after it loses a life or resets.
                # it can't be penalized if it doesn't resummon the ball... :)
                action = 1  
            else:
                action = self.model.MakeMoveTest(env.state)

            _, _, done, didLoseLife = env.step(action)

        print("Finished Testing!")

    def Learn(self):
        """
        Description:
            Performs backpropagation and updates the gradients on the main model
        """
        observs, actions, rewards, dones, nextObservs = self.memory.Sample(self.batchSize)

        maxQ = self.model.Predict(nextObservs).argmax(axis=1)

        futureQ = self.targetModel.Predict(nextObservs)
        doubleQ = futureQ[range(self.batchSize), maxQ]

        targetQ = (rewards + (0.95 * doubleQ * (1-dones))).astype('float32')

        with tf.GradientTape() as tape:
            currentQ = self.model.model(observs)  # we use model(observs) instead of model.predict(observs) because this is differentiable
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.numActions, dtype=np.float32)
            Q = tf.reduce_sum(tf.multiply(currentQ, one_hot_actions), axis=1)

            h_loss = keras.losses.Huber()(targetQ, Q)

        modelGrads = tape.gradient(h_loss, self.model.model.trainable_variables)
        self.model.model.optimizer.apply_gradients(zip(modelGrads, self.model.model.trainable_variables))
        
        return float(h_loss.numpy())



    def Train(self, numSteps=5000000):
        """
        Description:
            Runs the gym environment
        Parameters:
            numSteps (int)       - The number of frames to train for
        Returns:
            None
        """
        print("Training...")

        frameNum = 0
        while frameNum < numSteps:
            env.reset()
            didLoseLife = True
            for _ in range(MAX_GAME_FRAMES):
                action = self.model.MakeMove(env.state, frameNum, doUpdateExploreRate=(frameNum > self.startBatchSize))
                observation, reward, done, didLoseLife = env.step(action)

                self.memory.Remember(observation[:, :, 0], action, reward, didLoseLife)

                frameNum += 1

                if frameNum > self.startBatchSize:
                    if frameNum % self.trainTime == 0:
                        loss = self.Learn() # TODO: Aryan this is where you get the Huber loss per train

                    if frameNum % self.updateTime == 0:
                        self.Update() # update target network weights
                
                if done:
                    done = False
                    break
            
                if frameNum % 500000 == 0:
                    print(f"Frame: {frameNum}")

        print("Done Training!")

    def SaveModel(self):
        self.model.Save()


######################################################################################################
#                                           Main
######################################################################################################
if __name__ == '__main__':
    startTime = datetime.now()
    env = SetupEnvironment()

    print(f"\nAction space size: {env.env.action_space.n}")
    print(f"Actions: {env.env.unwrapped.get_action_meanings()}\n")

    if DO_RUN_TEST:
        agent = DuelingDQNAgent(env, doTest=True)
        agent.Test()
    else: # train the model then save it
        agent = DuelingDQNAgent(env, doTest=False)
        agent.Train()
        agent.SaveModel()

    env.close()

    print(f"Duration: {datetime.now() - startTime}")
