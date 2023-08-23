import gym
from gym import Env
from gym.spaces import Discrete, Box, Tuple
import random
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt

render_scale = 30  # MUST BE EVEN INTEGER

grid_rows = 10
grid_cols = 10
start_location = (0,0)
end_location = (9,9)
obstacles = ((1,0),(1,1),(1,2),(1,3),(2,3),(4,4),(4,5),(4,6),(4,7))
max_steps = 100

actions = {
    0 : (0,1),  # North
    1 : (0,-1), # South
    2 : (1,0),  # East
    3 : (-1,0), # West
    4 : (1,1),  # North-East
    5 : (1,-1), # North-West
    6 : (-1,1), # South-East
    7 : (-1,-1) # South-West
}

class State():

    def __init__(self):

        #Actions we can take: N, S, E, W
        self.action_space = Discrete(len(actions))
        # 2D Grid
        self.observation_space = Discrete(grid_rows*grid_cols)
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
            
    
    def render(self):

        image = np.ones((grid_rows*render_scale, grid_cols*render_scale, 3), dtype=np.uint8) * 255

        #image = cv2.putText(image,"Hello", org=(render_scale,render_scale),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 ,color=(0,0,0), thickness=2)

        h, w, _ = image.shape
        #rows, cols = (4,4)
        dy, dx = h / grid_rows, w / grid_cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=grid_cols-1):
            x = int(round(x))
            cv2.line(image, (x, 0), (x, h), color=(0, 0, 0), thickness=1)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=grid_rows-1):
            y = int(round(y))
            cv2.line(image, (0, y), (w, y), color=(0, 0, 0), thickness=1)



        image = cv2.rectangle(img=image,
            pt1=(int(end_location[0]*render_scale), int(end_location[1])*render_scale),
            pt2=(int(end_location[0]*render_scale+render_scale), int(end_location[1])*render_scale+render_scale),
            color=(0, 128, 0),
            thickness=-1)

        image = cv2.rectangle(img=image,
            pt1=(int(start_location[0]*render_scale), int(start_location[1])*render_scale),
            pt2=(int(start_location[0]*render_scale+render_scale), int(start_location[1])*render_scale+render_scale),
            color=(225, 165, 0),
            thickness=-1)

        for obstacle in obstacles:
            image = cv2.rectangle(img=image,
                pt1=(int(obstacle[0]*render_scale), int(obstacle[1])*render_scale),
                pt2=(int(obstacle[0]*render_scale+render_scale), int(obstacle[1])*render_scale+render_scale),
                color=(255, 0, 255),
                thickness=-1)
            
        pos_x, pos_y = self.state   
        image = cv2.circle(img=image,
            center=((int((pos_x)*render_scale)+int(0.5*render_scale)), (int((pos_y)*render_scale)+int(0.5*render_scale))),
            radius=int(0.5*render_scale),
            color=(255, 0, 0),
            thickness=-1)
        
        cv2.imshow("GridEnv2D", image)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            return
    def reset(self):
        self.state = start_location
        self.max_steps = max_steps
