import gym
from gym import Env
from gym.spaces import Discrete, Box, Tuple
import random
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt

grid_rows = 4
grid_cols = 4
start_location = (0,0)
end_location = (3,3)
obstacles = ((1,0),(2,3))
max_steps = 100

actions = {
    0 : (0,1),  # North
    1 : (0,-1), # South
    2 : (1,0),  # East
    3 : (-1,0)  # West
}

class State():

    def __init__(self):

        #Actions we can take: N, S, E, W
        self.action_space = Discrete(4)
        # 2D Grid
        self.observation_space = Discrete(16)
        # Set start location
        self.state = start_location
        # Set max steps
        self.max_steps = max_steps
        print("Agent start location is:",self.state)

    def check(self):
        print(self.state[0])

    def step(self, action):

        # Apply action
        self.state_new = tuple([sum(x) for x in zip(self.state,actions[action])])
        reward = 0
        
        # check if Agent moved out of bounds
        if self.state_new[0] not in range(0,grid_rows) or self.state_new[1] not in range(0,grid_cols):
            self.state = self.state
            print("Agent moved out of bounds. Position reset.")

        # check if Agent travelled through an obstacle
        elif self.state_new in obstacles:
            self.state = self.state_new 
            reward = -11
            print("Agent travelled through obstacle.")

        else:
            self.state = self.state_new      
            print("Agent moved to new position")


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
            
    
    def render(self):

        image = np.ones((400, 400, 3), dtype=np.uint8) * 255

        #image = cv2.putText(image,"Hello", org=(100,100),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 ,color=(0,0,0), thickness=2)

        h, w, _ = image.shape
        rows, cols = (4,4)
        dy, dx = h / rows, w / cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            cv2.line(image, (x, 0), (x, h), color=(0, 0, 0), thickness=1)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            cv2.line(image, (0, y), (w, y), color=(0, 0, 0), thickness=1)



        image = cv2.rectangle(img=image,
            pt1=(int(end_location[0]*100), int(end_location[1])*100),
            pt2=(int(end_location[0]*100+100), int(end_location[1])*100+100),
            color=(0, 128, 0),
            thickness=-1)

        image = cv2.rectangle(img=image,
            pt1=(int(start_location[0]*100), int(start_location[1])*100),
            pt2=(int(start_location[0]*100+100), int(start_location[1])*100+100),
            color=(225, 165, 0),
            thickness=-1)

        for obstacle in obstacles:
            image = cv2.rectangle(img=image,
                pt1=(int(obstacle[0]*100), int(obstacle[1])*100),
                pt2=(int(obstacle[0]*100+100), int(obstacle[1])*100+100),
                color=(255, 0, 255),
                thickness=-1)
            
        pos_x, pos_y = self.state   
        image = cv2.circle(img=image,
            center=((int(pos_x)*100)+50, (int(pos_y)*100)+50),
            radius=50,
            color=(255, 0, 0),
            thickness=-1)
        
        cv2.imshow("GridEnv2D", image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            return
    def reset(self):
        self.state = start_location
        self.max_steps = max_steps