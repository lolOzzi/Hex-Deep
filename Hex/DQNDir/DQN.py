import time
import random
import threading
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, BatchNormalization, 
                                     ReLU, Add, Concatenate, Input, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from TB import ModifiedTensorBoard
from HexEnv import *
from DQNNFD import NoisyFactorisedDense

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "5x5-Hybrid-PER-Optimized"
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 128
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

PER_e = 0.01
PER_a = 0.6
PER_b = 0.4
PER_b_increment_per_sampling = 0.001

class SumTree:
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
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.beta = PER_b

    def __len__(self):
        return self.tree.n_entries

    def store(self, experience):
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, experience)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + PER_b_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += PER_e
        clipped_errors = np.minimum(abs_errors, 1.0)
        ps = np.power(clipped_errors, PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class GINConv(tf.keras.layers.Layer):
    def __init__(self, hidden_units, **kwargs):
        super(GINConv, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.mlp = tf.keras.Sequential([
            Dense(hidden_units, activation='relu'),
            Dense(hidden_units)
        ])

    def call(self, node_features, adjacency_matrix):
        adjacency_matrix = tf.cast(adjacency_matrix, dtype=node_features.dtype)
        aggregated_features = tf.matmul(adjacency_matrix, node_features)
        combined_features = aggregated_features + node_features
        return self.mlp(combined_features)

def create_hybrid_gnn_convnet_model(board_size=5):
    action_space_size = board_size * board_size + 1
    num_nodes = board_size * board_size

    conv_input = Input(shape=(board_size, board_size, 3), name="conv_input")
    node_input = Input(shape=(num_nodes, 3), name="node_input")
    adj_input = Input(shape=(num_nodes, num_nodes), name="adj_input")
    swap_flag_input = Input(shape=(1,), name="swap_flag_input")

    conv_layer = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_input)
    conv_layer = BatchNormalization()(conv_layer)
    res_layer = Conv2D(128, (3, 3), padding='same')(conv_layer)
    res_layer = BatchNormalization()(res_layer)
    res_layer = Add()([conv_layer, res_layer])
    res_layer = ReLU()(res_layer)
    flat_conv_output = Flatten()(res_layer)

    gnn_layer = GINConv(128)(node_input, adj_input)
    gnn_layer = GINConv(128)(gnn_layer, adj_input)
    graph_embedding = GlobalAveragePooling1D()(gnn_layer)

    fused_layer = Concatenate()([flat_conv_output, graph_embedding, swap_flag_input])
    
    noisy_dense_layer = NoisyFactorisedDense(512)(fused_layer)
    rectified_ndl = ReLU()(noisy_dense_layer)
    noisy_dense_layer = NoisyFactorisedDense(256)(rectified_ndl)
    rectified_ndl = ReLU()(noisy_dense_layer)
    output_q_values = NoisyFactorisedDense(action_space_size, name="q_values", dtype='float32')(rectified_ndl)

    model = Model(inputs=[conv_input, node_input, adj_input, swap_flag_input], outputs=output_q_values)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
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
        self.replay_buffer_lock = threading.Lock()
        
        if train:
            self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
            self.tensorboard2 = ModifiedTensorBoard(log_dir=f"logs/P2-{MODEL_NAME}-{int(time.time())}")
            self.dataset_iterator = self._create_dataset_iterator()

        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        with self.replay_buffer_lock:
            self.replay_memory.store(transition)

    def _replay_generator(self):
        while True:
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                time.sleep(0.1)
                continue

            with self.replay_buffer_lock:
                minibatch, indices, is_weights = self.replay_memory.sample(MINIBATCH_SIZE)
            
            states, actions, rewards, next_states, dones = zip(*minibatch)

            current_conv_inputs, current_node_inputs, current_adj_inputs, current_swap_flags = zip(*states)
            next_conv_inputs, next_node_inputs, next_adj_inputs, next_swap_flags = zip(*next_states)
            
            current_states_tuple = (
                np.array(current_conv_inputs, dtype=np.float32),
                np.array(current_node_inputs, dtype=np.float32),
                np.array(current_adj_inputs, dtype=np.float32),
                np.array(current_swap_flags, dtype=np.float32)
            )
            next_states_tuple = (
                np.array(next_conv_inputs, dtype=np.float32),
                np.array(next_node_inputs, dtype=np.float32),
                np.array(next_adj_inputs, dtype=np.float32),
                np.array(next_swap_flags, dtype=np.float32)
            )
            actions_np = np.array(actions, dtype=np.int32)
            rewards_np = np.array(rewards, dtype=np.float32)
            dones_np = np.array(dones, dtype=np.bool)
            is_weights_np = np.array(is_weights, dtype=np.float32)
            indices_np = np.array(indices, dtype=np.int32)

            yield (current_states_tuple, actions_np, rewards_np, next_states_tuple, 
                   dones_np, is_weights_np, indices_np)

    def _create_dataset_iterator(self):
        
        board_size = self.env.SIZE
        num_nodes = board_size * board_size

        dataset = tf.data.Dataset.from_generator(
            self._replay_generator,
            output_signature=(
                (tf.TensorSpec(shape=(MINIBATCH_SIZE, board_size, board_size, 3), dtype=tf.float32),
                 tf.TensorSpec(shape=(MINIBATCH_SIZE, num_nodes, 3), dtype=tf.float32),
                 tf.TensorSpec(shape=(MINIBATCH_SIZE, num_nodes, num_nodes), dtype=tf.float32),
                 tf.TensorSpec(shape=(MINIBATCH_SIZE, 1), dtype=tf.float32)),
                tf.TensorSpec(shape=(MINIBATCH_SIZE,), dtype=tf.int32),
                tf.TensorSpec(shape=(MINIBATCH_SIZE,), dtype=tf.float32),
                (tf.TensorSpec(shape=(MINIBATCH_SIZE, board_size, board_size, 3), dtype=tf.float32),
                 tf.TensorSpec(shape=(MINIBATCH_SIZE, num_nodes, 3), dtype=tf.float32),
                 tf.TensorSpec(shape=(MINIBATCH_SIZE, num_nodes, num_nodes), dtype=tf.float32),
                 tf.TensorSpec(shape=(MINIBATCH_SIZE, 1), dtype=tf.float32)),
                tf.TensorSpec(shape=(MINIBATCH_SIZE,), dtype=tf.bool),
                tf.TensorSpec(shape=(MINIBATCH_SIZE,), dtype=tf.float32),
                tf.TensorSpec(shape=(MINIBATCH_SIZE,), dtype=tf.int32)
            )
        )
        return iter(dataset.prefetch(tf.data.AUTOTUNE))
        
    @tf.function
    def get_qs(self, conv_input, node_input, adj_input, swap_flag_input):
        inputs = [
            tf.expand_dims(conv_input, axis=0),
            tf.expand_dims(node_input, axis=0),
            tf.expand_dims(adj_input, axis=0),
            tf.expand_dims(swap_flag_input, axis=0)
        ]
        q_values = self.model(inputs, training=False)
        return q_values[0]

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones, is_weights):
        future_q_values = self.target_model(next_states, training=False)
        max_future_q = tf.reduce_max(future_q_values, axis=1)
        target_q_values = rewards + (1.0 - tf.cast(dones, tf.float32)) * DISCOUNT * (-max_future_q)

        with tf.GradientTape() as tape:
            all_q_values = self.model(states, training=True)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            predicted_q_values = tf.gather_nd(all_q_values, action_indices)
            
            loss_object = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            per_element_loss = loss_object(target_q_values, predicted_q_values)
            loss = tf.reduce_mean(is_weights * per_element_loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        abs_errors = tf.abs(target_q_values - predicted_q_values)
        return abs_errors

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        batch_data = next(self.dataset_iterator)
        states, actions, rewards, next_states, dones, is_weights, indices = batch_data

        abs_errors = self.train_step(states, actions, rewards, next_states, dones, is_weights)
        
        with self.replay_buffer_lock:
            self.replay_memory.batch_update(indices.numpy(), abs_errors.numpy())
        
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