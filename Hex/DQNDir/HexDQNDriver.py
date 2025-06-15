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


model_file = 'models/5x5-simple_____5.00max____2.70avg____0.00min__1749991339.keras'
#model_file = None


def loadModel(model_file_path):

    if model_file_path:
        print(f"Loading model from: {model_file_path}")
        try:
            agent.model = tf.keras.models.load_model(
                model_file_path,
                custom_objects={'NoisyFactorisedDense': NoisyFactorisedDense}
            )
            agent.target_model.set_weights(agent.model.get_weights())
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
    else:
        print("No model file found, starting from scratch.")


loadModel(model_file)

# Environment settings
SIZE = 5

EPISODES = 900_000
SELF_PLAY_START_EPISODE = 2000
MOVE_PENALTY_DECAY_EPISODE = 0
MOVE_PENALTY_BASE_VALUE = -0.05
MOVE_PENALTY_DECAY_VALUE = 0
#MOVE_PENALTY_DECAY_VALUE = 0

MIN_REWARD = -5 - (SIZE*SIZE // 2) * MOVE_PENALTY_BASE_VALUE

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes
MODEL_SAVE_EVERY = 500
SHOW_PREVIEW = False

ep_rewards = [MIN_REWARD]
ep_rewards2 = [MIN_REWARD]

# For more repetitive results
random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)


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
        player = env.player_num
        all_q_values = agent.get_qs(current_state)

        valid_actions_mask_np = env.get_valid_actions_mask(player)
        valid_actions_mask_tensor = tf.convert_to_tensor(valid_actions_mask_np, dtype=tf.bool)
        masked_q_values = tf.where(
            valid_actions_mask_tensor,
            all_q_values,
            -np.inf
        )

        action_tensor = tf.argmax(masked_q_values)
        action = action_tensor.numpy()
        
        op_state_pre_action = env.getObservation(3 - player)
        cp_state_pre_action = env.getObservation(player)
        #  Execute action and process results
        if self_play:
            new_state, reward, done = env.step(action, player)

            op_state_post_action = env.getObservation(3-player)

            # Store state/action for potential loser punishment
            last_state_by_player[player]  = current_state
            last_action_by_player[player] = action

            if player == 1:  
                episode_reward += reward
            else:
                episode_reward2 += reward

            
            
            agent.update_replay_memory((current_state, action, reward, op_state_post_action, done))
            agent.train(done, step)

            # If the game ended, punish the loser
            if done:
                # Now identify the loser and inject a terminal loss for their last move.
                loser = 3 - player
                if last_state_by_player[loser] is not None:
                    loser_state  = last_state_by_player[loser]
                    loser_action = last_action_by_player[loser]
                    agent.update_replay_memory(
                        (loser_state, loser_action, env.LOSS_PENALTY, cp_state_pre_action, True)
                    )
                    agent.train(True, step)
            
            # Switch player and update state for the next turn
            env.player_num = 3 - player
            current_state = env.getObservation(env.player_num)
            step += 1

        else: # Play against a random opponent
            # Agent's move
            new_state, reward, done = env.step(action, env.player_num)
            episode_reward += reward
            
            if done: # Agent won 
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done, step)
                current_state = new_state
            else: # Game continues, random opponent's turn
                opponent_action_pos = env.calc_op_move()
                # We need to get the resulting state after the opponent moves
                env.hex.placeMove(opponent_action_pos, 3 - env.player_num)
                final_state = env.getObservation(env.player_num)
                final_state_op = env.getObservation(3 - env.player_num)
                if env.hex.checkWin(3 - env.player_num):
                    done = True
                    reward = env.LOSS_PENALTY # Agent lost
                    agent.update_replay_memory((current_state, action, reward, new_state, True))
                else:
                    agent.update_replay_memory((current_state, action, reward, final_state_op, False))
                # The 'new_state' for the agent's transition is after the opponent has moved

                agent.train(done, step)
                current_state = final_state

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
    if not episode % MODEL_SAVE_EVERY:
        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.keras')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)




