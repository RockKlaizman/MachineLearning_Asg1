import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes, is_training=True, render=False):
    env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=False,
        render_mode="human" if render else None,
    )
    # hyperparameters
    learning_rate = 0.9  # Learning Rate (α)
    discount_factor = 0.9  # discount factor (γ)
    epsilon = 1.0  # initial exploration rate
    epsilon_decay = 0.0001  # exploration rate decay
    rewards_table = np.zeros(episodes)

    if is_training:
        q_table = np.zeros([env.observation_space.n, env.action_space.n])
    else:
        with open("frozen_lake_q_table.pkl", "rb") as f:
            q_table = pickle.load(f)

    for i in range(episodes):
        state = env.reset()[0]
        failed = False
        finished = False

        while not finished and not failed:
            rnd = np.random.rand()
            if is_training and rnd < epsilon:
                action = (
                    env.action_space.sample()
                )  # 0 = left, 1 = down, 2 = right, 3 = up
            else:
                action = np.argmax(q_table[state, :])

            new_state, reward, finished, failed, _ = env.step(action)

            if failed:
                reward = -1
        

            if is_training:
                q_table[state, action] = q_table[state, action] + learning_rate * (
                    reward
                    + discount_factor * np.max(q_table[new_state, :])
                    - q_table[state, action]
                )
            

            state = new_state

        epsilon = max(epsilon - epsilon_decay, 0)

        if epsilon == 0:
            learning_rate = 0.0001

        rewards_table[i] = reward
        if i%1000 == 0 and is_training:
            print(f"Episode {i}/{episodes}")
    env.close()

    sum_rewards = np.zeros(episodes)
    for i in range(episodes):
        sum_rewards[i] = np.sum(rewards_table[max(0, i - 100) : i + 1])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    
    plt.imshow(q_table, cmap='hot', interpolation='nearest')
    plt.savefig('frozen_lake8x8_q_table.png')

    if is_training:
        with open("frozen_lake_q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)


if __name__ == "__main__":
    run(15000)
    #run(1, is_training=False, render=True)
