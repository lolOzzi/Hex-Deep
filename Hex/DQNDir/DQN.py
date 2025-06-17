import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, Add, Concatenate, BatchNormalization
import numpy as np
from TB import ModifiedTensorBoard
from HexEnv import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from DQNNFD import *
from PER import PERBuffer

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "5x5-simple-PER"
NORMALISATION_VALUE = 1
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.995
UPDATE_TARGET_EVERY = 5



class DQNAgent:
    def __init__(self, env, train=True):
        self.env = env
        self.randomGen = np.random.default_rng(2)
        if (len(tf.config.experimental.list_physical_devices('GPU')) > 0):
            configure_gpu_optimizations()
        # Primary Model, gets trained every step
        self.model = self.create_model()
        # Target Model, we .predict this one
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = PERBuffer(REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        if train:
            self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
            self.tensorboard2 = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}Player2-{int(time.time())}")
        
        self.lossfn = tf.keras.losses.MeanSquaredError()

    def create_model(self):
        """
        Basic DQN for 5x5 Hex
        """
        # --- Input Layer ---
        board_input = Input(shape=(5, 5, 2), name='board_input')
        swap_input = Input(shape=(1,), name='swap_input')

        # field: 3x3
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu',
                name='local_patterns')(board_input)
        
        # field: 5x5
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu',
                name='global_patterns')(x)
        
        # May not be needed, mayhelp with complex patterns
        x = Conv2D(128, kernel_size=3, padding='same', activation='relu',
                    name='pattern_combinations')(x)
        
        # Flatten for decision making
        x_flat = Flatten()(x)

        # --- Concatenation ---
        concatenated = Concatenate()([x_flat, swap_input])
        
        # Decision layers
        d = NoisyFactorisedDense(256, name='decision_layer_1')(concatenated)
        d = ReLU()(d)
        d = NoisyFactorisedDense(128, name='decision_layer_2')(d)
        d = ReLU()(d)
        
        # Output layer: Q-values for each position
        final_output = Dense(self.env.SIZE*self.env.SIZE, activation='linear', name='q_values')(d)
        model = Model(inputs=[board_input, swap_input], outputs=final_output, name='hex_dqn_5x5_noisy')
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        return model
        
    def update_replay_memory(self, transition):
        self.replay_memory.store(transition)
    
    @tf.function(
        input_signature=[
            (
                tf.TensorSpec(shape=(5, 5, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32)
            )
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
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones, is_weights):

        current_states_board, current_states_swap = states
        new_current_states_board, new_current_states_swap = next_states

        compute_dtype = self.model.compute_dtype
        rewards = tf.cast(rewards, dtype=compute_dtype)
        dones = tf.cast(dones, dtype=compute_dtype)
        is_weights = tf.cast(is_weights, dtype=compute_dtype)


        future_qs_list = self.target_model([new_current_states_board, new_current_states_swap], training=False)
        max_future_qs = tf.reduce_max(future_qs_list, axis=1)
        # future q's negative because its op pov
        target_q_values = rewards + (1.0 - dones) * DISCOUNT * (-max_future_qs)

        # Compute DQN Bellman‐error loss:
        #   target    = r + γ * max_a' Q_target(s', a')
        #   predicted = Q(s, a)
        #   loss      = mean( (target - predicted)^2 )
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(tf.cast(actions, tf.int32), self.env.ACTION_SPACE_SIZE, dtype=compute_dtype)
            q_values = self.model([current_states_board, current_states_swap], training=True)
            predicted_q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)

            element_wise_loss = tf.cast(self.lossfn(target_q_values, predicted_q_values), dtype=compute_dtype)
            weighted_loss = element_wise_loss * is_weights

            if isinstance(self.model.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                print("It is a lossscaleOptimizer")
                is_weights_cast = tf.cast(is_weights, dtype=scaled_loss.dtype)
                scaled_loss = self.model.optimizer.scale_loss(element_wise_loss)
                weighted_loss = scaled_loss * is_weights_cast
            loss = tf.reduce_mean(weighted_loss)
                        
            
        if isinstance(self.model.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.model.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)
        
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        abs_errors = tf.abs(target_q_values - predicted_q_values)
        return abs_errors

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        mini_batch, tree_indices, is_weights = self.replay_memory.sample(MINIBATCH_SIZE)

        # Unpack the batch and format for TensorFlow
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        def stack_components(pairs):
            board, swap = zip(*pairs)
            return np.array(board, dtype=np.float32), np.array(swap, dtype=np.float32)

        current_states_board, current_states_swap = stack_components(states)
        next_states_board, next_states_swap = stack_components(next_states)
        
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Perform a training step and get the errors
        abs_errors = self.train_step(
            (current_states_board, current_states_swap), 
            actions, 
            rewards, 
            (next_states_board, next_states_swap), 
            dones,
            is_weights
        )
        
        # Update the priorities in the replay buffer
        self.replay_memory.batch_update(tree_indices, abs_errors.numpy())


        # Periodically update the target model
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

def configure_gpu_optimizations():
    # Enable mixed precision 
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Configure GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.optimizer.set_jit(True)
    
    print("GPU optimizations enabled!")