import gym
env = gym.make("CartPole-v0")
obs = env.reset()
env.render()

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        env.render()
        if done:
            break
    totals.append(episode_rewards)

import numpy as np
print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))