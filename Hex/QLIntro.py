import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0")
env.reset()

# Funny constants
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 8_000

# Randomness / exploration constants
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_val = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


SHOW_EVERY = 2000
STATS_EVERY = 100

# stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

def get_discrete_state(state) :
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))



print(DISCRETE_OS_SIZE)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table)


for episode in range(EPISODES):
    episode_reward = 0

    render = episode % SHOW_EVERY == 0
    if render:
        print(episode)
    

    state = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT*max_future_q)

            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            #print(f"We did it on ep {episode}!")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_val
        
    ep_rewards.append(episode_reward)
    if episode % STATS_EVERY == 0:
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / len(ep_rewards[-STATS_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
    if episode % 10 == 0:
        np.save(f"QLTutQtables/{episode}-qtable.npy", q_table)


env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()