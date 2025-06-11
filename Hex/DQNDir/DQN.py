# DQN.py

import time
# Corrected imports: Changed from 'keras' to 'tensorflow.keras'
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
from TB import ModifiedTensorBoard
import random
from HexEnv import *
import tensorflow as tf
import os

# ... rest of the file remains the same

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "5x5-Hybrid-GNN-ConvNet-SwapFlag" # Updated Model Name
NORMALISATION_VALUE = 1
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 128
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
class GINConv(tf.keras.layers.Layer):
    """Graph Isomorphism Network (GIN) layer."""
    def __init__(self, hidden_units, **kwargs):
        super(GINConv, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.mlp = tf.keras.Sequential([
            Dense(hidden_units, activation='relu'),
            Dense(hidden_units)
        ])

    def call(self, node_features, adjacency_matrix):
        # Cast adjacency_matrix to match the dtype of node_features
        adjacency_matrix = tf.cast(adjacency_matrix, dtype=node_features.dtype)
        
        aggregated_features = tf.matmul(adjacency_matrix, node_features)
        combined_features = aggregated_features + node_features
        return self.mlp(combined_features)

def create_hybrid_gnn_convnet_model(board_size=5):
    """Creates the hybrid GNN and ConvNet model with swap_flag input."""
    action_space_size = board_size * board_size + 1
    num_nodes = board_size * board_size

    # Define the four inputs
    conv_input = Input(shape=(board_size, board_size, 3), name="conv_input")
    node_input = Input(shape=(num_nodes, 3), name="node_input")
    adj_input = Input(shape=(num_nodes, num_nodes), name="adj_input")
    swap_flag_input = Input(shape=(1,), name="swap_flag_input") # New input for the swap flag

    # Convolutional branch
    conv_layer = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_input)
    conv_layer = BatchNormalization()(conv_layer)
    res_layer = Conv2D(128, (3, 3), padding='same')(conv_layer)
    res_layer = BatchNormalization()(res_layer)
    res_layer = Add()([conv_layer, res_layer])
    res_layer = ReLU()(res_layer)
    flat_conv_output = Flatten()(res_layer)

    # GNN branch
    gnn_layer = GINConv(128)(node_input, adj_input)
    gnn_layer = GINConv(128)(gnn_layer, adj_input)
    graph_embedding = GlobalAveragePooling1D()(gnn_layer)

    # Fusion stage
    # Concatenate the outputs of all branches, including the new swap_flag_input
    fused_layer = Concatenate()([flat_conv_output, graph_embedding, swap_flag_input])
    
    # Dense layers for Q-value estimation
    dense_layer = Dense(512, activation='relu')(fused_layer)
    dense_layer = Dense(256, activation='relu')(dense_layer)
    output_q_values = Dense(action_space_size, activation='linear', name="q_values")(dense_layer)

    # Create and compile the model with four inputs
    model = Model(inputs=[conv_input, node_input, adj_input, swap_flag_input], outputs=output_q_values)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

class DQNAgent:
    def __init__(self, env, train=True):
        self.env = env
        if (len(tf.config.experimental.list_physical_devices('GPU')) > 0):
            configure_gpu_optimizations()
        
        self.model = create_hybrid_gnn_convnet_model(self.env.SIZE)
        self.target_model = create_hybrid_gnn_convnet_model(self.env.SIZE)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.randomGen = np.random.default_rng(1)
        
        if train:
            self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
            self.tensorboard2 = ModifiedTensorBoard(log_dir=f"logs/P2-{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(HexEnv.SIZE, HexEnv.SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(HexEnv.SIZE * HexEnv.SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(HexEnv.SIZE * HexEnv.SIZE, HexEnv.SIZE * HexEnv.SIZE), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32)
    ])
    def get_qs(self, conv_input, node_input, adj_input, swap_flag_input):
        """
        Get Q-values for a single state. Compiles into a TF graph.
        Note: The input is four separate tensors, not a list.
        """
        # The model expects a batch dimension, so we add it.
        inputs = [
            tf.expand_dims(conv_input, axis=0),
            tf.expand_dims(node_input, axis=0),
            tf.expand_dims(adj_input, axis=0),
            tf.expand_dims(swap_flag_input, axis=0)
        ]
        # Call the model directly (more efficient in a tf.function)
        q_values = self.model(inputs, training=False)
        return q_values[0] # Return the Q-values for the single state

    @tf.function(input_signature=[
        (tf.TensorSpec(shape=(None, HexEnv.SIZE, HexEnv.SIZE, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, HexEnv.SIZE * HexEnv.SIZE, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, HexEnv.SIZE * HexEnv.SIZE, HexEnv.SIZE * HexEnv.SIZE), dtype=tf.float32),
         tf.TensorSpec(shape=(None, 1), dtype=tf.float32)),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        (tf.TensorSpec(shape=(None, HexEnv.SIZE, HexEnv.SIZE, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, HexEnv.SIZE * HexEnv.SIZE, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, HexEnv.SIZE * HexEnv.SIZE, HexEnv.SIZE * HexEnv.SIZE), dtype=tf.float32),
         tf.TensorSpec(shape=(None, 1), dtype=tf.float32)),
        tf.TensorSpec(shape=(None,), dtype=tf.bool)
    ])
    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            future_q_values = self.target_model(next_states, training=False)
            max_future_q = tf.reduce_max(future_q_values, axis=1)
            
            target_q = rewards + (1 - tf.cast(dones, tf.float32)) * DISCOUNT * max_future_q
            mask = tf.one_hot(tf.cast(actions, dtype=tf.int32), self.env.ACTION_SPACE_SIZE)
            q_values = self.model(states, training=True)
            predicted_q = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            
            loss = tf.keras.losses.MSE(target_q, predicted_q)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Define the data signature for a single state (now with 4 components)
        state_signature = (
            tf.TensorSpec(shape=(self.env.SIZE, self.env.SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(self.env.SIZE * self.env.SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(self.env.SIZE * self.env.SIZE, self.env.SIZE * self.env.SIZE), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32) # Added signature for swap_flag
        )

        # Create a TensorFlow dataset from the minibatch with the updated signature
        dataset = tf.data.Dataset.from_generator(
            lambda: minibatch,
            output_signature=(
                state_signature, # state
                tf.TensorSpec(shape=(), dtype=tf.int32), # action
                tf.TensorSpec(shape=(), dtype=tf.float32), # reward
                state_signature, # next_state
                tf.TensorSpec(shape=(), dtype=tf.bool) # done
            )
        )

        dataset = dataset.batch(MINIBATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        for states, actions, rewards, next_states, dones in dataset:
            self.train_step(states, actions, rewards, next_states, dones)

        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

def configure_gpu_optimizations():
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)