import os
import gym
import time
import random
import pickle
import numpy as np
import pandas as pd
from itertools import chain
from collections import deque
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
import matplotlib.pyplot as plt

plt.style.use("seaborn")
plt.rcParams["font.family"] = "IPAexGothic"

class DQN(object):

    def __init__(self, env_id, agent_hist_len=4, memory_size=2000,
                 replay_start_size=32, gamma=0.99, eps=1.0, eps_min=1e-4,
                 final_expl_step=1000, mb_size=32, C=100, n_episodes=400,
                 max_steps=500):

        self.env_id = env_id
        self.env = gym.make(env_id)
        self.path = "./data/" + env_id
        self.agent_hist_len = agent_hist_len
        self.memory_size = memory_size
        self.replay_start_size = replay_start_size
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.final_expl_step = final_expl_step
        self.eps_decay = (eps-eps_min) / final_expl_step
        self.mb_size = mb_size
        self.C = C
        self.n_episodes = n_episodes
        self.max_steps = max_steps

        self._init_memory()
        self.scaler = StandardScaler()
        self.scaler.fit(np.array([t[0] for t in self.memory]))

    @staticmethod
    def _flatten_deque(d):
        return np.array(list(chain(*d)))

    def _get_optional_action(self, network, agent_hist):
        agent_hist_normalized = self.scaler.transform(
            self._flatten_deque(agent_hist).reshape(1, -1))
        return np.argmax(network.predict(agent_hist_normalized)[0])

    def _get_action(self, agent_hist=None):
        if agent_hist is None:
            return self.env.action_space.sample()
        else:
            self.eps = max(self.eps - self.eps_decay, self.eps_min)
            if np.random.random() < self.eps:
                return self.env.action_space.sample()
            else:
                return self._get_optional_action(self.Q, agent_hist)

    def _remember(self, agent_hist, action, reward, new_state, done):
        self.memory.append([self._flatten_deque(agent_hist), action, reward,
                            new_state if not done else None])

    def _init_memory(self):
        print("Initializing replay memory: ", end="")
        self.memory = deque(max_len=self.memory_size)
        while True:
            state = self._get_action(agent_hist=None)
            new_state, reward, done, _ = self.env.step(action)
            if len(agent_hist) == self.agent_hist_len:
                self._remember(agent_hist, action, reward, new_state, done)
            if len(self.memory) == self.replay_start_size:
                print("done")
                return
            if done:
                break
            state = new_state
            agent_hist.append(state)

    def _build_network(self):
        nn = Sequential()
        nn.add(Dense(20, activation="relu",
                     input_dim=(self.agent_hist_len
                                * self.env.observation_space.shape[0])))
        nn.add(Dense(20, activation="relu"))
        nn.add(Dense(10, activation="relu"))
        nn.add(Dense(self.env.action_space.n))
        nn.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return nn

    def _clone_network(self, nn):
        clone = self._build_network()
        clone.set_weights(nn.get_weights())
        return clone

    def _get_samples(self):
        samples = random.sample(self.memory, self.mb_size)
        agent_hists = np.array([s[0] for s in samples])
        Y = self.target_Q.predict(self.scaler.transform(agent_hists))
        actions = [s[1] for s in samples]
        rewards = np.array([s[2] for s in samples])
        future_rewards = np.zeros(self.mb_size)
        new_states_idx = [i for i, s in enumerate(samples) if s[3] is not None]
        new_states = np.array([s[3] for s in itemgetter(*new_states_idx)(samples)])
        new_agent_hists = np.hstack(
            [agent_hists[new_states_idx, self.env.observation_space.shape[0]:],
             new_states])
        future_rewards[new_states_idx] = np.max(
            self.target_Q.predict(self.scaler.transform(new_agent_hists)), aixs=1)
        rewards += self.gamma*future_rewards
        for i, r in enumerate(Y):
            Y[i, actions[i]] = rewards[i]
        return agent_hists, Y

    def _replay(self):
        agent_hists, Y = self._get_samples()
        agent_hists_normalized = self.scaler.transform(agent_hists)
        for i in range(self.mb_size):
            self.Q.train_on_batch(agent_hists_normalized[i, :].reshape(1, -1),
                                  Y[i, :].reshape(1, -1))

    def learn(self, render=False, verbose=True):

        self.Q = self._build_network()
        self.target_Q = self._clone_network(self.Q)

        if verbose:
            print("Learning target network:")
        self.scores = []
        for episode in range(self.n_episodes):
            state = self.env.reset()
            agent_hist = deque(maxlen=self.agent_hist_len)
            agent_hist.append(state)
            score = 0
            for step in range(self.max_steps):
                if render:
                    self.env.render()
                if len(agent_hist) < self.agent_hist_len:
                    action = self._get_action(agent_hist=None)
                else:
                    action = self._get_action(agent_hist)
                new_state, reward, done, _ = self.env.step(action)
                if verbose:
                    print("episode: {:4} | step: {:3} | memory: {:6} | \
                           eps: {:.4f} | action: {} | reward: {:.1f} | best score: {:6.1f} | \
                           mean score: {:6.1f}".format(
                                                    episode+1, step+1, len(self.memory), self.eps, action, reward,
                                                    max(self.scores) if len(self.scores) != 0 else np.nan,
                                                    np.mean(self.scores) if len(self.scores) != 0 else np.nan),
                                                    end="\r")
                score += reward
                if len(agent_hist) == self.agent_hist_len:
                    