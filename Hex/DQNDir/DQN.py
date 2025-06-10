import time
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, Add, Concatenate
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
MODEL_NAME = "5x5-mars-s-c+d-res"
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

    
    def residual_block(self, x, filters):
        """
        Defines a single residual block with two convolutional layers and a skip connection.
        This structure helps in training deeper networks by mitigating vanishing gradients.

        Args:
            x: The input tensor to the block.
            filters (int): The number of filters for the convolutional layers.

        Returns:
            The output tensor of the residual block.
        """
        # Store the input tensor for the skip connection 
        shortcut = x

        # First convolutional path
        y = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)

        # Second convolutional path
        y = Conv2D(filters, kernel_size=(3, 3), padding='same')(y)
        y = BatchNormalization()(y)

        # Add the shortcut to the output of the convolutional paths 
        y = Add()([shortcut, y])
        # Final activation after the addition
        y = ReLU()(y)

        return y

    def create_model(self):
        """
        Creates the Deep Q-Network (DQN) model for a 5x5 Hex board.
        The architecture is a deep residual network inspired by AlphaZero.

        Returns:
            A TensorFlow/Keras model.
        """
            # --- Input Layer ---
        # The input is a 5x5 board with 3 channels: player's stones, opponent's stones,
        # and a channel indicating the location of P1's first move for a potential swap.
        board_input = Input(shape=(5, 5, 3), name='board_input')
        
        # A separate input for a single flag (1.0 or 0.0) indicating if the swap is available.
        swap_input = Input(shape=(1,), name='swap_input')

        # --- Convolutional Body ---
        # This shared body processes the spatial information of the board.
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(board_input)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # 4 residual blocks to learn deep features.
        for _ in range(4):
            x = self.residual_block(x, filters=128)

        # --- Head 1: Q-Values for Board Moves ---
        # This head outputs the Q-values for placing a piece on any of the 25 cells.
        board_q_head = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                            activation='linear', name='board_q_values')(x)
        # Flatten the (5, 5, 1) output to (25,) to represent Q-values for each cell.
        board_q_flat = Flatten(name='board_q_flat')(board_q_head)

        # --- Head 2: Q-Value for the Swap Action ---
        # This head determines the Q-value of performing the swap.
        # It uses the flattened features from the convolutional body and the swap availability flag.
        swap_head_features = Flatten()(x)
        swap_head_features = Concatenate()([swap_head_features, swap_input])
        swap_head = Dense(128, activation='relu')(swap_head_features)
        swap_q_value = Dense(1, activation='linear', name='swap_q_value')(swap_head)

        # --- Final Concatenation ---
        # Combine the Q-values from both heads into a single output tensor of shape (26,).
        # This matches the environment's action space (25 board moves + 1 swap move).
        final_output = Concatenate(name='final_q_values')([board_q_flat, swap_q_value])

        # --- Create and Compile Model ---
        model = Model(inputs=[board_input, swap_input], outputs=final_output, name='hex_dqn_5x5_swap')
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