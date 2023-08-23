import gym
from gym import Env
from gym.spaces import Discrete, Box, Tuple
import random
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt

cube_rows = 6
cube_cols = 6
cube_height = 6

start_location = (0,0,0)
end_location = (5,5,5)

obstacles = ((5,5,4),(5,4,5))

max_steps = 5000

actions = {
    0 : (0,1,0),  # North
    1 : (0,-1,0), # South
    2 : (1,0,0),  # East
    3 : (-1,0,0), # West
    4 : (1,1,0),  # North-East
    5 : (1,-1,0), # North-West
    6 : (-1,1,0), # South-East
    7 : (-1,-1,0), # South-West
    8 : (0,0,1),  # Up
    9 : (0,0,-1)  # Down
}

class State():

    def __init__(self):

        #Actions we can take: N, S, E, W
        self.action_space = Discrete(len(actions))
        # 3D cube
        self.observation_space = Discrete(cube_rows*cube_cols*cube_height)
        # Set start location
        self.state = start_location
        # Set max steps
        self.max_steps = max_steps
        print("Agent start location is:",self.state)
        self.obstacles = obstacles
        self.xlim = cube_rows
        self.ylim = cube_cols
        self.zlim = cube_height

    def check(self):
        print(self.state[0])

    def step(self, action):

        # Apply action
        self.state_new = tuple([sum(x) for x in zip(self.state,actions[action])])
        reward = 0

        # check if Agent moved out of bounds
        if self.state_new[0] not in range(0,cube_rows) or \
            self.state_new[1] not in range(0,cube_cols) or \
            self.state_new[2] not in range(0,cube_height):
            self.state = self.state
            # print("Agent moved out of bounds. Position reset.")

        # check if Agent travelled through an obstacle
        elif self.state_new in obstacles:
            self.state = self.state_new 
            reward = -11
            # print("Agent travelled through obstacle.")

        else:
            self.state = self.state_new      
            # print("Agent moved to new position")


        self.max_steps -= 1

        # Check if Agent reached goal location
        if self.state == end_location:
            reward = 1
            done = True
        # check if max steps reached
        elif self.max_steps <= 0:
            done = True
        else:
            done = False

        info = {}
        return self.state, reward, done, info
    
    # def render(self):
    
    def reset(self):
        self.state = start_location
        self.max_steps = max_steps