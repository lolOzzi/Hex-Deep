import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from game import *
hex = Hex_Game(5)
# Funny constants
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 50_000

# Randomness / exploration constants
epsilon = 1
epsilon_min = 0.1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_val = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
END_RANDOM_OPPONENT = EPISODES
random_op_move = True

SHOW_EVERY = 25000
STATS_EVERY = 100

# stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}


    
def state_to_key(board: np.ndarray) -> tuple:
    return tuple(board.reshape(-1).tolist())


q_table = defaultdict(lambda: np.random.uniform(low=-2, high=0, size=hex.BOARD_SIZE*hex.BOARD_SIZE))
empty = np.zeros((hex.BOARD_SIZE,hex.BOARD_SIZE), dtype=int)
q_vals = q_table[state_to_key(empty)]
print(q_table)

def swap_board():
    arr = hex.board.copy()
    arr[arr == 1] = -1
    arr[arr == 2] = 1 
    arr[arr == -1] = 2
    return arr.T

def calc_op_move():
    return random.choice(hex.actionspace)
    ''' For later
    if random_op_move:
        return random.choice(hex.actionspace)
    
    q_table[state_to_key(swap_board())]
    opp_q_vals = q_table[state_to_key(swap_board())]
    valid_flat_actions = [i * hex.board.shape[1] + j for (i, j) in hex.actionspace]
    valid_q_values = [(a, opp_q_vals[a]) for a in valid_flat_actions]
    return divmod(max(valid_q_values, key=lambda x: x[1])[0], hex.board.shape[1])'''


def hex_step(move):
    move = divmod(move, hex.board.shape[1])
    hex.placeMove(move, 1)
    reward = -1
    done = False
    if (hex.checkWin(1)) :
        reward = 10
        #print("nowayyyy")
        return state_to_key(hex.board), reward, True
    hex.placeMove(calc_op_move(), 2)
    if (hex.checkWin(2)) :
        reward = -10
        return state_to_key(hex.board), reward, True
    return state_to_key(hex.board), reward, done





for episode in range(EPISODES):
    episode_reward = 0

    render = episode % SHOW_EVERY == 0
    if render:
        print(episode)
    
    random_op_move = episode < END_RANDOM_OPPONENT

    state = state_to_key(empty)
    hex.reset()

    done = False
    while not done:
        state_key = tuple(hex.board.flatten())
        all_q_values = q_table[state_key]  
        
        valid_flat_actions = [i * hex.board.shape[1] + j for (i, j) in hex.actionspace]

        if np.random.rand() < epsilon:
            action = np.random.choice(valid_flat_actions)
        else:
            valid_q_values = [(a, all_q_values[a]) for a in valid_flat_actions]
            action = max(valid_q_values, key=lambda x: x[1])[0]
        
        new_state_key, reward, done = hex_step(action)
        episode_reward += reward

        if render:
            hex.hex_grid(size=1, dims=hex.BOARD_SIZE)

        if not done:
            max_future_q = np.max(q_table[new_state_key])
            current_q = q_table[state_key][action]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT*max_future_q)

            q_table[state_key][action] = new_q
        elif hex.checkWin(1) or hex.checkWin(2):
            #print(f"We did it on ep {episode}!")
            q_table[state_key][action] = reward

        state = new_state_key

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_val
        epsilon = max(epsilon_min, epsilon)
        
    ep_rewards.append(episode_reward)
    if episode % STATS_EVERY == 0:
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / len(ep_rewards[-STATS_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
    if episode % 10 == 0:
        np.save(f"HexQtables/{episode}-qtable.npy", dict(q_table), allow_pickle=True)



plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()