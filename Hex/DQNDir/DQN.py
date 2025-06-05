import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import numpy as np
from collections import deque
from TB import *
import random
from HexEnv import *
from DQN import *
import numpy as np
import tensorflow.python.keras.backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import models
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "256x2"
NORMALISATION_VALUE = 1 # maybe not, 255 if rgb.
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
class DQNAgent:

    def __init__(self, env):

        self.env = env
        
        # Primary Model, get trained every step
        self.model = self.create_model()
        

        #Target Model, we .predict this one
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        


    def create_model(self):


        inputs = Input(shape=self.env.OBSERVATION_SPACE_VALUES)

        # Convolutional trunk
        x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)

        # Flatten and fully-connected head
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(self.env.ACTION_SPACE_SIZE, activation='linear')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name='hex_dqn')
        model.compile(loss="mse", optimizer='Adam', metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def get_qs(self, state):
            x = state.astype(np.float32) / NORMALISATION_VALUE
            return self.model.predict(x[np.newaxis], verbose=0)[0]
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/NORMALISATION_VALUE
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/NORMALISATION_VALUE
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(X)/NORMALISATION_VALUE, np.array(y), batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        # Incrementing and checking if we want to update target_model
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

