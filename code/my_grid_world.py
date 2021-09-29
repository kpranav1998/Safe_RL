import numpy as np
import random

class GridWorld():
    def set(self):
        self.state = np.zeros((4,4))
        self.player = None ## 1 ##
        self.pit = (1,2) ## 2 ##
        self.goal = (2,3) ## 3 ##
        self.randPair()
        
        self.state[self.player] = 1
        self.state[self.pit] = 2
        self.state[self.goal] = 3


    def randPair(self):
        while(self.player == None):
            state = (random.randint(0,3),random.randint(0,3))
            if(state != self.pit and state != self.goal):
                self.player = state



    
    def getReward(self):
        player_loc = self.player
        if(player_loc == self.pit):
            return -10
        elif(player_loc == self.goal):
            return 10
        else:
            return -1

    def next(self,action):

        actions = [[-1,0],[1,0],[0,-1],[0,1]]  # up, down, left, right #
        new_state = np.zeros((4,4))
        next_position = (self.player[0] + actions[action][0],self.player[1] + actions[action][1])
        

        if(next_position[0] <= 3 and next_position[0] >= 0):
            if(next_position[1] <= 3 and next_position[1] >= 0):
                new_state[next_position] = 1
                new_state[self.pit] = 2
                new_state[self.goal] = 3
                self.state = new_state
                self.player = next_position
        return self.state

    def display(self):
        return self.state
        














