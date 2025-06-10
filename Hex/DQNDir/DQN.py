import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Concatenate
from keras.optimizers import Adam
import numpy as np
from collections import deque
from TB import ModifiedTensorBoard
import random
from HexEnv import *
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
import os
from PIL import Image
import cv2

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "5x5-tellus-s-c+d-3l"
NORMALISATION_VALUE = 1  # maybe not, 255 if rgb.
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 128
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
class DQNAgent:
    def __init__(self, env, train=True):
        self.env = env
        configure_rtx5070ti()
        # Primary Model, gets trained every step
        self.model = self.create_model()
        # Target Model, we .predict this one
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        if train:
            self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
            self.tensorboard2 = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}Player2-{int(time.time())}")
        self.target_update_counter = 0
        self.randomGen = np.random.default_rng(1)

    

    def create_model(self):
        # --- Input Layers ---
        # Input for the 5x5 board state (3 channels: player pieces, opponent pieces, swappable piece)
        board_input = Input(shape=(self.env.SIZE, self.env.SIZE, 3), name='board_input')
        
        # Input for the single swap-availability flag
        swap_input = Input(shape=(1,), name='swap_input')
        '''5x5 Venus''' 

        # Layer 1: Local pattern detection (3x3 receptive field)
        # 64 filters to capture various local Hex patterns
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu',
                name='local_patterns')(board_input)
        
        # Layer 2: Global pattern detection (5x5 receptive field = full board)
        # 128 filters to capture board-wide strategic patterns
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu',
                name='global_patterns')(x)
        
        # Optional: One more layer for pattern combinations
        # Only if you see improvement in training
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu',
                    name='pattern_combinations')(x)
        
        # Flatten for decision making
        x_flat = Flatten()(x)

        # --- Concatenation ---
        # Merge the flattened board features with the swap flag
        concatenated = Concatenate()([x_flat, swap_input])
        
        # Decision layers
        d = Dense(256, activation='relu', name='decision_layer_1')(concatenated)
        d = Dropout(0.2)(d)  # Prevent overfitting
        d = Dense(128, activation='relu', name='decision_layer_2')(d)
        
        # Output layer: Q-values for each position
        outputs = Dense(self.env.ACTION_SPACE_SIZE, activation='linear', name='q_values')(d)

        ''' Hex 11x11?
        # Convolutional trunk
        x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        # Flatten and fully-connected head
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(self.env.ACTION_SPACE_SIZE, activation='linear')(x)
        '''
        model = Model(inputs=[board_input, swap_input], outputs=outputs, name='hex_dqn')
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    @tf.function(
        input_signature=[
            [
                tf.TensorSpec(shape=(5, 5, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32)
            ]
        ]
    )
    def get_qs(self, state):
        board_state, swap_flag = state
        board_state = board_state
        swap_flag = swap_flag

        # Predict using a list of inputs
        return self.model(
            [board_state[np.newaxis, ...], swap_flag[np.newaxis, ...]], training=False
        )[0]
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        sampled_indices = self.randomGen.choice(len(self.replay_memory), MINIBATCH_SIZE, replace=False)

        # Create the minibatch by retrieving the transitions using the sampled indices.
        # This is a fast list comprehension.
        minibatch = [self.replay_memory[i] for i in sampled_indices]

        # Deconstruct states into separate arrays for board and swap flags
        current_states_board = np.array([transition[0][0] for transition in minibatch])
        current_states_swap = np.array([transition[0][1] for transition in minibatch])

        new_current_states_board = np.array([transition[3][0] for transition in minibatch])
        new_current_states_swap = np.array([transition[3][1] for transition in minibatch])

        # Predict Q-values in batches
        current_qs_list = self.model.predict([current_states_board, current_states_swap], verbose=0, batch_size=MINIBATCH_SIZE)
        future_qs_list = self.target_model.predict([new_current_states_board, new_current_states_swap], verbose=0, batch_size=MINIBATCH_SIZE)

        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Vectorized Bellman Equation
        max_future_qs = np.max(future_qs_list, axis=1)
        new_q_values = rewards + (1 - dones) * DISCOUNT * max_future_qs

        # Update the Q-values for the actions taken
        target_qs_list = current_qs_list
        rows_to_update = np.arange(MINIBATCH_SIZE)
        target_qs_list[rows_to_update, actions] = new_q_values

        # Fit the model on the entire prepared batch
        self.model.fit(
            [current_states_board, current_states_swap],
            target_qs_list,
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False
        )
        
        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

def configure_rtx5070ti():
    # Enable mixed precision (RTX 5070 Ti loves this)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Configure GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Enable XLA compilation
    tf.config.optimizer.set_jit(True)
    
    print("RTX 5070 Ti optimizations enabled!")