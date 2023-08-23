import gym
from gym import Env
from gym.spaces import Discrete, Box, Tuple
import random
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt

from GridWorld2D_env import *

class Agent:

    def __init__(self):
        self.states = []
        self.State = State()
        self.actions = list(actions.keys())
        self.lr = 0.2
        self.exp_rate = 0.9
        self.decay_gamma = 0.9   

        #initlialize Q table

        self.Qvalues = {}   
        for i in range(grid_rows):
            for j in range(grid_cols):
                self.Qvalues[(i,j)] = {}
                for a in self.actions:
                    self.Qvalues[(i,j)][a] = 0

    def chooseAction(self):
        mx_nxt_rwd = 0
        action = ""

        if np.random.uniform(0,1) <= self.exp_rate:
            action = np.random.choice(self.actions)
            # print("EXPLORE")
        else:
            action = max(self.Qvalues[self.State.state])
            for a in self.actions:
                current_pos = self.State.state
                nxt_reward = self.Qvalues[current_pos][a]
                if nxt_reward >= mx_nxt_rwd:
                    action = a
                    mx_nxt_rwd = nxt_reward

            # print("Action chosen:",action)
            # print("Q-value of chosen action",mx_nxt_rwd)
            # print("EXPLOIT")
        return action
    
    def train(self, episodes = 10):
        self.scores = []
        self.max_score = -5000
        for episode in range(1, episodes+1):
            self.State.reset()
            state = start_location
            done = False
            score = 0
            moves = 0
            locations = []
            max_epsilon = 1.0             # Exploration probability at start
            min_epsilon = 0.05            # Minimum exploration probability 
            decay_rate = 0.0005            # Exponential decay rate for exploration prob
            self.exp_rate = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
            
            while not done:
                #self.State.render()
                # print("Step", moves)
                action = self.chooseAction()
                # take action
                n_state, reward, done, info = self.State.step(action)
                # print(n_state)
                old_value = self.Qvalues[state][action]
                next_max = max(self.Qvalues[n_state].values())
                # print(next_max)
                new_value = (1 - self.lr) * old_value + self.lr *(reward + self.decay_gamma * next_max)
                self.Qvalues[state][action] = new_value
                state = n_state
                locations.append(n_state)
                score += reward
                moves += 1
                
            print('Episode:{} Score:{} Moves:{} Eps Greedy:{}'.format(episode,\
                                             score, moves, self.exp_rate))
            self.scores.append(score)
            if score > self.max_score:
                    self.max_score = score
                    self.best_locations = locations
            # self.State.render()
            # cv2.waitKey(1000) # waits until a key is pressed
            # cv2.destroyAllWindows() # destroys the window showing image
        self.learned_Qvalues = self.Qvalues

    def test(self, episodes = 10):

        # same process as train() except Q values are not updated, and all actions are greedy

        self.scores = []
        self.final_route = []
        self.max_score = -5000
        for episode in range(1, episodes+1):
            self.State.reset()
            state = start_location
            done = False
            score = 0
            moves = 0
            locations = []
            
            while not done:
                self.State.render()
                print("Step", moves)
                action = self.chooseAction()
                #print("Chosen action:",action,type(action))
                print("current position {} action {}".format(state, action))
                n_state, reward, done, info = self.State.step(action)
                print(n_state)                
                state = n_state
                locations.append(n_state)
                score += reward
                moves += 1
            print('Episode:{} Score:{} Moves:{} Locations:{} Eps Greedy:{}'.format(episode,\
                                             score, moves, locations,self.exp_rate))
            self.scores.append(score)
            if score > self.max_score:
                    self.max_score = score
                    self.final_route = locations
            self.State.render()
            cv2.waitKey(500) # waits until a key is pressed
            cv2.destroyAllWindows() # destroys the window showing image

                 
