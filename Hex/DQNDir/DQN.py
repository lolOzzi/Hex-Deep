# DQN.py

import time
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, Add, Concatenate, BatchNormalization

import numpy as np
from collections import deque
from TB import ModifiedTensorBoard
from HexEnv import *
import tensorflow as tf
import os
from tensorflow.keras.losses import MeanSquaredError 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from DQNNFD import *

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "5x5-Hybrid"
NORMALISATION_VALUE = 1
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 128
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

# --- Prioritized Experience Replay Parameters ---
PER_e = 0.01  # Epsilon: small value added to the priority to ensure no transition has zero priority.
PER_a = 0.6   # Alpha: determines how much prioritization is used. (0: uniform sampling, 1: full prioritization)
PER_b = 0.4   # Beta: importance-sampling exponent, anneals to 1.0.
PER_b_increment_per_sampling = 0.001

class SumTree:
    """
    A binary tree data structure where the value of a parent node is the sum of its children.
    This allows for efficient O(log n) sampling of items based on their priorities.
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class ReplayBuffer:
    """
    A Prioritized Experience Replay buffer that uses a SumTree for efficient sampling.
    """
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.beta = PER_b

    def __len__(self):
        return self.tree.n_entries

    def store(self, experience):
        # New experiences are given the highest priority to ensure they are trained on.
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, experience)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        # Anneal beta towards 1.0
        self.beta = np.min([1., self.beta + PER_b_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Calculate importance-sampling weights to correct for the biased sampling.
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def batch_update(self, tree_idx, abs_errors):
        # Update the priorities of the sampled transitions based on their TD-error.
        abs_errors += PER_e  # Add epsilon to ensure non-zero priority
        clipped_errors = np.minimum(abs_errors, 1.0) # Clip errors to 1 for stability
        ps = np.power(clipped_errors, PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


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
    noisy_dense_layer = NoisyFactorisedDense(512)(fused_layer)
    rectified_ndl = ReLU()(noisy_dense_layer)
    noisy_dense_layer = NoisyFactorisedDense(256)(rectified_ndl)
    rectified_ndl = ReLU()(noisy_dense_layer)
    output_q_values = NoisyFactorisedDense(action_space_size, name="q_values")(rectified_ndl)

    # Create and compile the model with four inputs
    model = Model(inputs=[conv_input, node_input, adj_input, swap_flag_input], outputs=output_q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=MeanSquaredError())
    return model


class DQNAgent:
    def __init__(self, env, train=True):
        self.env = env
        if (len(tf.config.experimental.list_physical_devices('GPU')) > 0):
            configure_gpu_optimizations()
        
        self.model = create_hybrid_gnn_convnet_model(self.env.SIZE)
        self.target_model = create_hybrid_gnn_convnet_model(self.env.SIZE)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = ReplayBuffer(REPLAY_MEMORY_SIZE)

        self.randomGen = np.random.default_rng(1)
        
        if train:
            self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
            self.tensorboard2 = ModifiedTensorBoard(log_dir=f"logs/P2-{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0


    def update_replay_memory(self, transition):
        self.replay_memory.store(transition)
        
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
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    ])
    def train_step(self, states, actions, rewards, next_states, dones, is_weights):
        # Calculate target Q-values using the Double DQN update rule
        future_q_values = self.target_model(next_states, training=False)
        max_future_q = tf.reduce_max(future_q_values, axis=1)
        max_future_q = tf.cast(max_future_q, tf.float32)
        target_q_values = rewards + (1.0 - tf.cast(dones, tf.float32)) * DISCOUNT * max_future_q

        with tf.GradientTape() as tape:
            # Get the Q-values for the taken actions using the main model
            all_q_values = self.model(states, training=True)
            
            # Gather the Q-values corresponding to the actions taken
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            predicted_q_values = tf.gather_nd(all_q_values, action_indices)

            # Calculate per-element loss for priority updates
            abs_errors = tf.abs(target_q_values - predicted_q_values)
            
            # Use a loss object that doesn't perform reduction
            loss_object = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            per_element_loss = loss_object(target_q_values, predicted_q_values)

            # Apply importance-sampling weights and reduce to a single loss value
            loss = tf.reduce_mean(is_weights * per_element_loss)

        # Calculate gradients and apply them
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return abs_errors


    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch, indices, is_weights = self.replay_memory.sample(MINIBATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Correctly unpack the 4-part states
        current_conv_inputs, current_node_inputs, current_adj_inputs, current_swap_flags = zip(*states)
        next_conv_inputs, next_node_inputs, next_adj_inputs, next_swap_flags = zip(*next_states)

        # Prepare state tuples for the train_step function
        current_states_tuple = (
            tf.convert_to_tensor(np.array(current_conv_inputs), dtype=tf.float32),
            tf.convert_to_tensor(np.array(current_node_inputs), dtype=tf.float32),
            tf.convert_to_tensor(np.array(current_adj_inputs), dtype=tf.float32),
            tf.convert_to_tensor(np.array(current_swap_flags), dtype=tf.float32)
        )
        next_states_tuple = (
            tf.convert_to_tensor(np.array(next_conv_inputs), dtype=tf.float32),
            tf.convert_to_tensor(np.array(next_node_inputs), dtype=tf.float32),
            tf.convert_to_tensor(np.array(next_adj_inputs), dtype=tf.float32),
            tf.convert_to_tensor(np.array(next_swap_flags), dtype=tf.float32)
        )
        actions_tensor = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
        rewards_tensor = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(np.array(dones), dtype=tf.bool)
        is_weights_tensor = tf.convert_to_tensor(is_weights, dtype=tf.float32)
        
        # Perform a training step and get the absolute errors for the batch
        abs_errors = self.train_step(
            current_states_tuple,
            actions_tensor,
            rewards_tensor,
            next_states_tuple,
            dones_tensor,
            is_weights_tensor
        )
        
        # Update priorities in the replay buffer
        self.replay_memory.batch_update(indices, abs_errors.numpy())
        
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