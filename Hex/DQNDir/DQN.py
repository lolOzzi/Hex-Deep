import time
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Input
from keras.optimizers import Adam
import numpy as np
from collections import deque
from TB import ModifiedTensorBoard
import random
from HexEnv import *
import tensorflow as tf
import os

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "5x5-Hybrid-GNN-ConvNet"
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
        aggregated_features = tf.matmul(adjacency_matrix, node_features)
        combined_features = aggregated_features + node_features
        return self.mlp(combined_features)

def create_hybrid_gnn_convnet_model(board_size=5):
    """Creates the hybrid GNN and ConvNet model."""
    action_space_size = board_size * board_size + 1
    num_nodes = board_size * board_size

    conv_input = Input(shape=(board_size, board_size, 3), name="conv_input")
    node_input = Input(shape=(num_nodes, 3), name="node_input")
    adj_input = Input(shape=(num_nodes, num_nodes), name="adj_input")

    conv_layer = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_input)
    conv_layer = BatchNormalization()(conv_layer)
    res_layer = Conv2D(128, (3, 3), padding='same')(conv_layer)
    res_layer = BatchNormalization()(res_layer)
    res_layer = Add()([conv_layer, res_layer])
    res_layer = ReLU()(res_layer)
    flat_conv_output = Flatten()(res_layer)

    gnn_layer = GINConv(128)(node_input, adj_input)
    gnn_layer = GINConv(128)(gnn_layer, adj_input)
    graph_embedding = tf.reduce_mean(gnn_layer, axis=1)

    fused_layer = Concatenate()([flat_conv_output, graph_embedding])
    dense_layer = Dense(512, activation='relu')(fused_layer)
    dense_layer = Dense(256, activation='relu')(dense_layer)
    output_q_values = Dense(action_space_size, activation='linear', name="q_values")(dense_layer)

    model = Model(inputs=[conv_input, node_input, adj_input], outputs=output_q_values)
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
        
        if train:
            self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
            self.tensorboard2 = ModifiedTensorBoard(log_dir=f"logs/P2-{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(
            [np.array([state[0]]), np.array([state[1]]), np.array([state[2]])],
            verbose=0
        )[0]

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            # Get Q-values for the next states from the target model
            future_q_values = self.target_model(next_states, training=False)
            max_future_q = tf.reduce_max(future_q_values, axis=1)
            
            # Calculate the target Q-value
            target_q = rewards + (1 - dones) * DISCOUNT * max_future_q

            # Get the Q-values for the current states and actions from the primary model
            mask = tf.one_hot(tf.cast(actions, dtype=tf.int32), self.env.ACTION_SPACE_SIZE)
            q_values = self.model(states, training=True)
            predicted_q = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            
            # Calculate loss
            loss = tf.keras.losses.MSE(target_q, predicted_q)
            
        # Backpropagate loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Unpack the minibatch
        current_states = [transition[0] for transition in minibatch]
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = [transition[3] for transition in minibatch]
        dones = np.array([transition[4] for transition in minibatch])
        
        # Prepare states for TensorFlow
        current_conv_states = np.array([s[0] for s in current_states])
        current_node_states = np.array([s[1] for s in current_states])
        current_adj_states = np.array([s[2] for s in current_states])

        next_conv_states = np.array([s[0] for s in next_states])
        next_node_states = np.array([s[1] for s in next_states])
        next_adj_states = np.array([s[2] for s in next_states])
        
        self.train_step(
            [current_conv_states, current_node_states, current_adj_states],
            actions,
            rewards,
            [next_conv_states, next_node_states, next_adj_states],
            dones
        )

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