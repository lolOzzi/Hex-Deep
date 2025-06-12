import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, Add, Concatenate, BatchNormalization
import numpy as np
from collections import deque
from TB import ModifiedTensorBoard
from HexEnv import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from DQNNFD import *

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "5x5-jupiter"
NORMALISATION_VALUE = 1  # maybe not, 255 if rgb.
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 128
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5



class DQNAgent:
    def __init__(self, env, train=True):
        self.env = env
        if (len(tf.config.experimental.list_physical_devices('GPU')) > 0):
            configure_gpu_optimizations()
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
        self.randomGen = np.random.default_rng(2)
        self.lossfn = tf.keras.losses.MeanSquaredError()

    
    def residual_block(self, x, filters, block_num):
        """
        Defines a single residual block with two convolutional layers and a skip connection.
        This structure helps in training deeper networks by mitigating vanishing gradients.
        """
        shortcut = x

        # First convolutional path
        y = Conv2D(filters, kernel_size=(3, 3), padding='same', name=f'res{block_num}_conv1')(x)
        y = BatchNormalization(name=f'res{block_num}_bn1')(y)
        y = ReLU(name=f'res{block_num}_relu1')(y)

        # Second convolutional path
        y = Conv2D(filters, kernel_size=(3, 3), padding='same', name=f'res{block_num}_conv2')(y)
        y = BatchNormalization(name=f'res{block_num}_bn2')(y)

        # Add the shortcut to the output of the convolutional paths 
        y = Add(name=f'res{block_num}_add')([shortcut, y])
        # Final activation after the addition
        y = ReLU(name=f'res{block_num}_relu2')(y)

        return y
    
    def create_old_model(self):
        board_input = Input(shape=(5, 5, 3), name='board_input')
        swap_input = Input(shape=(1,), name='swap_input')

        # --- Convolutional Body (with explicit names) ---
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='initial_conv')(board_input)
        x = BatchNormalization(name='initial_bn')(x)
        x = ReLU(name='initial_relu')(x)

        # 4 residual blocks to learn deep features.
        # The loop now passes the block number to the helper function.
        for i in range(4):
            x = self.residual_block(x, filters=128, block_num=i)

        # --- Head 1: Q-Values for Board Moves ---
        board_q_head = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                            activation='linear', name='board_q_values')(x)
        board_q_flat = Flatten(name='board_q_flat')(board_q_head)

        # --- Head 2: Q-Value for the Swap Action ---
        swap_head_features = Flatten(name='swap_flatten')(x)
        swap_head_features = Concatenate(name='swap_concat')([swap_head_features, swap_input])
        swap_head = Dense(128, activation='relu', name='swap_dense_1')(swap_head_features)
        swap_q_value = Dense(1, activation='linear', name='swap_q_value')(swap_head)

        # --- Final Concatenation ---
        final_output = Concatenate(name='final_q_values')([board_q_flat, swap_q_value])

        # --- Create and Compile Model ---
        model = Model(inputs=[board_input, swap_input], outputs=final_output, name='hex_dqn_5x5_swap')
        #model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        
        return model
    
    def create_model(self):
        """
        Creates the Deep Q-Network (DQN) model for a 5x5 Hex board.
        The architecture is a deep residual network inspired by AlphaZero.
        The final dense layers are replaced with Noisy Layers.
        """
        # --- Input Layer ---
        board_input = Input(shape=(5, 5, 3), name='board_input')
        swap_input = Input(shape=(1,), name='swap_input')

        # --- Convolutional Body (ADD EXPLICIT NAMES) ---
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='initial_conv')(board_input)
        x = BatchNormalization(name='initial_bn')(x)
        x = ReLU(name='initial_relu')(x)
        
        # Also add names to the layers inside the residual blocks
        for i in range(4):
            x = self.residual_block(x, filters=128, block_num=i)

        # --- Head 1: Q-Values for Board Moves ---
        board_q_head = Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                            activation='linear', name='board_q_values')(x)
        board_q_flat = Flatten(name='board_q_flat')(board_q_head)

        # Concat with swap 
        swap_head_features = Flatten(name='swap_flatten')(x) # Name this flatten layer as well
        swap_head_features = Concatenate()([swap_head_features, swap_input])
        
        # Noisy layers
        swap_head = NoisyFactorisedDense(128, name='noisy_dense_1')(swap_head_features)
        swap_head = ReLU()(swap_head)
        swap_q_value = NoisyFactorisedDense(1, name='swap_q_value')(swap_head) # Output layer

        # Combine and compile
        final_output = Concatenate(name='final_q_values',  dtype='float32')([board_q_flat, swap_q_value])


        model = Model(inputs=[board_input, swap_input], outputs=final_output, name='hex_dqn_5x5_noisy')
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        
        return model
        



    def load_partial_weights(self, old_model_path):
        """
        Loads weights from an old model into a new NoisyNet model,
        transferring only the weights of layers that match by name.
        """
        print("Loading weights from old model...")
        
        # 1. Create an instance of the old model architecture
        old_model = self.create_old_model()
        
        # 2. Load the weights from your file into the old architecture
        old_model.load_weights(old_model_path)
        
        # 3. Transfer weights layer by layer
        layers_transferred = 0
        for layer in self.model.layers:
            try:
                # Find layer with the same name in the old model
                old_layer = old_model.get_layer(name=layer.name)
                
                # Get and set weights
                self.model.get_layer(name=layer.name).set_weights(old_layer.get_weights())
                self.target_model.get_layer(name=layer.name).set_weights(old_layer.get_weights())
                print(f"  - Transferred weights for layer: {layer.name}")
                layers_transferred += 1
            except ValueError:
                # This will happen for the NoisyFactorisedDense layers, which is expected
                print(f"  - Skipped layer (no match or different type): {layer.name}")

        print(f"\nSuccessfully transferred weights for {layers_transferred} layers.")
        if layers_transferred == 0:
            print("Warning: No layers were transferred. Check if layer names are consistent between models.")


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    @tf.function(
        input_signature=[
            (
                tf.TensorSpec(shape=(5, 5, 3), dtype=tf.float32),
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
    
    @tf.function(
        input_signature=(
            # The 'states' and 'next_states' tuples now have a 'None' for the batch dimension.
            (tf.TensorSpec(shape=(None, 5, 5, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, 1), dtype=tf.float32)),
            
            # Actions, rewards, and dones are now vectors of size 'None' (the batch size).
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            
            (tf.TensorSpec(shape=(None, 5, 5, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, 1), dtype=tf.float32)),
            
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    def train_step(self, states, actions, rewards, next_states, dones, norm=False):
        """
        Performs a single, highly optimized training step.
        This function is compiled into a static graph.
        """
        # Unpack states, which are tuples of (board, swap_flag)
        current_states_board, current_states_swap = states
        new_current_states_board, new_current_states_swap = next_states

        future_qs_list = self.target_model([new_current_states_board, new_current_states_swap], training=False)
        max_future_qs = tf.reduce_max(future_qs_list, axis=1)
        if norm:
            target_q_values = rewards + (1.0 - dones) * DISCOUNT * max_future_qs
        else:
            #Negated, becuase q's are for opponent.
            target_q_values = rewards + (1.0 - dones) * DISCOUNT * (-max_future_qs)

        # Compute DQN Bellman‐error loss:
        #   target    = r + γ * max_a' Q_target(s', a')
        #   predicted = Q(s, a)
        #   loss      = mean( (target - predicted)^2 )
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(tf.cast(actions, tf.int32), self.env.ACTION_SPACE_SIZE)
            q_values = self.model([current_states_board, current_states_swap], training=True)
            predicted_q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = self.lossfn(target_q_values, predicted_q_values)
            
            # Loss scaled, for use in mixed precesion calcs
            scaled_loss = self.model.optimizer.get_scaled_loss(loss)
        
        # Calculate gradients using the scaled loss.
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # Unscale the gradients back to their original magnitude before applying them.
        gradients = self.model.optimizer.get_unscaled_gradients(scaled_gradients)
        
        # Apply the unscaled gradients to the model's variables.
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def train(self, terminal_state, step):
        """
        Orchestrates training using a high-performance tf.data input pipeline.
        """
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Sample a minibatch from the replay memory
        minibatch_indices = self.randomGen.choice(len(self.replay_memory), MINIBATCH_SIZE, replace=False)
        minibatch = [self.replay_memory[i] for i in minibatch_indices]

        # Unzip the minibatch
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Separate board and swap_flag from states and next_states
        current_states_board, current_states_swap = zip(*states)
        new_current_states_board, new_current_states_swap = zip(*next_states)

        # Convert to tensors
        current_states_board = tf.convert_to_tensor(np.array(current_states_board), dtype=tf.float32)
        current_states_swap = tf.convert_to_tensor(np.array(current_states_swap), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
        rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
        new_current_states_board = tf.convert_to_tensor(np.array(new_current_states_board), dtype=tf.float32)
        new_current_states_swap = tf.convert_to_tensor(np.array(new_current_states_swap), dtype=tf.float32)
        dones = tf.convert_to_tensor(np.array(dones), dtype=tf.float32)

        # Call the train_step function with the prepared tensors
        self.train_step(
            (current_states_board, current_states_swap),
            actions,
            rewards,
            (new_current_states_board, new_current_states_swap),
            dones
        )

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
    
    # Enable XLA compilation
    tf.config.optimizer.set_jit(True)
    
    print("RTX 5070 Ti optimizations enabled!")