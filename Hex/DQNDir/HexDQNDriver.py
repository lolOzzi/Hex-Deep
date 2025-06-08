from HexEnv import *
from DQN import *
import numpy as np
import tensorflow as tf
import time
import random
from tqdm import tqdm

print("Heres the gpu info", tf.config.list_physical_devices('GPU'))
env = HexEnv()
agent = DQNAgent(env)

#model_file = 'models/4lconv128b____-2.00max___-5.99avg___-9.00min__1749285701.keras'
model_file = None
if model_file:
    agent.model.load_weights(model_file)
    agent.target_model.set_weights(agent.model.get_weights())
    print("Model loaded")
else:
    print("No modelfile found")

SIZE = 5
MIN_REWARD = -5 - (SIZE*SIZE // 2) 

# Environment settings
EPISODES = 20_000
SELF_PLAY_START_EPISODE = 1000
MOVE_PENALTY_DECAY_EPISODE = 1000
MOVE_PENALTY_BASE_VALUE = -0.1
MOVE_PENALTY_DECAY_VALUE = -0.05

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes
SHOW_PREVIEW = False

ep_rewards = [MIN_REWARD]
ep_rewards2 = [MIN_REWARD]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode
    agent.tensorboard2.step = episode

    episode_reward = 0
    episode_reward2 = 0
    step = 1

    done = False
        # Check if self-play mode should be enabled
    self_play = episode >= SELF_PLAY_START_EPISODE
    current_state = env.reset() if self_play else env.resetRand()

    last_state_by_player = {1: None, 2: None}
    last_action_by_player = {1: None, 2: None}

    env.turn_penalty = MOVE_PENALTY_BASE_VALUE \
                       if episode < MOVE_PENALTY_DECAY_EPISODE \
                       else MOVE_PENALTY_DECAY_VALUE

    while not done:
        if self_play:

            # Agent plays as both players
            player = env.player_num
            all_q_values = agent.get_qs(current_state)

            if (player==1):
                valid_flat_actions = [i * env.SIZE + j for (i, j) in env.hex.actionspace]
            else:
                valid_flat_actions = [i * env.SIZE + j for (j, i) in env.hex.actionspace]
            
            # Exploration vs exploitation
            if np.random.random() < epsilon:
                action = agent.randomGen.choice(valid_flat_actions)
            else:
                valid_q_values = [(a, all_q_values[a]) for a in valid_flat_actions]
                action = max(valid_q_values, key=lambda x: x[1])[0]
            
            row, col = divmod(action, SIZE)
            

            # If player 2, the board was transposed, so the action must be un-transposed
            corrected_move = (row, col)
            if player == 2 :
                corrected_move = (col, row) 
            new_state, reward, done = env.step(corrected_move, player)


            #Store moves for later
            last_state_by_player[player]  = current_state
            last_action_by_player[player] = action
            current_state = new_state


            if player == 1:  
                episode_reward += reward
            else:
                episode_reward2 += reward
            # Store transition and train
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            # Punish the loser
            if done:
                # 2) now figure out who the loser is
                loser = 3 - player

                # you need to know what the loser’s *last* action & state were…
                # so earlier in the loop you should have kept track of:
                #     last_state_by_player = {1: None, 2: None}
                #     last_action_by_player = {1: None, 2: None}
                #
                # and updated them immediately after each env.step call:
                #     last_state_by_player[player]  = current_state
                #     last_action_by_player[player] = action

                loser_state  = last_state_by_player[loser]
                loser_action = last_action_by_player[loser]

                # 3) inject a “terminal loss” for that losing side
                agent.update_replay_memory(
                    (loser_state,
                    loser_action,
                    env.LOSS_PENALTY,    # explicit negative terminal reward
                    new_state,           # terminal observation (could be same for both)
                    True)                # done=True
                )
                agent.train(True, step)  # train on that transition, too
                break

            
            # Switch player and update state
            current_state = env.getObservation(3 - player)  # Switch to other player (1 -> 2, 2 -> 1)
            env.player_num = 3 - player
            step += 1
        else:
            # Play against random opponent
            all_q_values = agent.get_qs(current_state)
            if (env.player_num==1):
                valid_flat_actions = [i * env.SIZE + j for (i, j) in env.hex.actionspace]
            else:
                valid_flat_actions = [i * env.SIZE + j for (j, i) in env.hex.actionspace]
            
            if np.random.random() < epsilon:
                action = agent.randomGen.choice(valid_flat_actions)
            else:
                valid_q_values = [(a, all_q_values[a]) for a in valid_flat_actions]
                action = max(valid_q_values, key=lambda x: x[1])[0]
            
            row, col = divmod(action, SIZE)
            corrected_move = (row, col)
            if env.player_num == 2 :
                corrected_move = (col, row) 
            new_state, reward, done = env.step(corrected_move, env.player_num)
            episode_reward += reward
            episode_reward2 += reward
            
            if not done:
                # Random opponent's move
                opponent_action = env.calc_op_move()
                env.hex.placeMove(opponent_action, 3 - env.player_num)
                if env.hex.checkWin(3 - env.player_num):
                    done = True
                    reward = env.LOSS_PENALTY
            
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)
            current_state = new_state
            step += 1

    
     # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    ep_rewards2.append(episode_reward2)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        average_reward2 = sum(ep_rewards2[-AGGREGATE_STATS_EVERY:])/len(ep_rewards2[-AGGREGATE_STATS_EVERY:])
        min_reward2 = min(ep_rewards2[-AGGREGATE_STATS_EVERY:])
        max_reward2 = max(ep_rewards2[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard2.update_stats(reward_avg=average_reward2, reward_min=min_reward2, reward_max=max_reward2, epsilon=epsilon)
        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD or min_reward2 >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.keras')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)




