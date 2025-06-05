from HexEnv import *
from DQN import *
import numpy as np
import tensorflow.python.keras.backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

print("Heres the gpu info", tf.config.list_physical_devices('GPU'))
env = HexEnv()
agent = DQNAgent(env)

SIZE = 5
MIN_REWARD = -5 - (SIZE*SIZE // 2) 

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes
SHOW_PREVIEW = False

ep_rewards = [MIN_REWARD]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False
    while not done:

        all_q_values = agent.get_qs(current_state)
        
        valid_flat_actions = [i * env.hex.board.shape[1] + j for (i, j) in env.hex.actionspace]

        if np.random.random() < epsilon:
            action = np.random.choice(valid_flat_actions)
        else:
            valid_q_values = [(a, all_q_values[a]) for a in valid_flat_actions]
            action = max(valid_q_values, key=lambda x: x[1])[0]

        flat = action
        row, col = divmod(flat, SIZE)
        new_state, reward, done = env.step((row, col))

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()
        
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    
     # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.keras')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)




