import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()    #added

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        # Training
        if (self._train):

            # Update Q table UNLESS it is your first time
            if (self.s != None and self.a != None):
                # Extract vars
                head_x = state[0]
                head_y = state[1]
                body   = state[2]
                food_x = state[3]
                food_y = state[4]

                # Set rewards based on previous state and action
                if (dead):
                    reward = -1
                elif (points-self.points):
                    reward = 1
                else:
                    reward = -0.1

                # Update Q
                n_s = set_state(state)      # New state
                ALPHA = self.C / (self.C + self.N[self.s][self.a])
                self.Q[self.s][self.a] += (ALPHA*(reward + self.gamma*find_highest_Q(self.Q, n_s) - self.Q[self.s][self.a]))

            else:
                n_s = set_state(state)

            # Check if last move killed ya
            if dead:
                self.reset()
                return 0

            # Find next action -- accounts for ties, priority 3210
            _max = (float)("-inf")
            for d in range(3, -1, -1):
                n = self.N[n_s][d]
                q = self.Q[n_s][d]
                # Exploration
                if n < self.Ne:
                    n_a = d             # New action
                    break
                # Exploitation
                elif q > _max:
                    _max = q
                    n_a = d

            # Update N table with that action
            self.N[n_s][n_a] += 1

            # Update self vars
            self.s = n_s
            self.a = n_a
            self.points = points

            return n_a

        # Testing
        else:
            return find_next_action(self.Q, set_state(state))
       

def find_next_action(Q, s):
    return np.argmax(Q[s][:])

def find_highest_Q(Q, s):
    return np.amax(Q[s][:])

def set_state(state):
    snake_head_x = state[0]
    snake_head_y = state[1]
    snake_body = state[2]
    food_x = state[3]
    food_y = state[4]

    if (snake_head_x == 40):                            adjoining_wall_x = 1        # Wall on left
    elif (snake_head_x == 480):                         adjoining_wall_x = 2        # Wall on right
    else:                                               adjoining_wall_x = 0

    if (snake_head_y == 40):                            adjoining_wall_y = 1        # Wall on top
    elif (snake_head_y == 480):                         adjoining_wall_y = 2        # Wall on bottom
    else:                                               adjoining_wall_y = 0

    if (snake_head_x > food_x):                         food_dir_x = 1              # Food on left
    elif (snake_head_x < food_x):                       food_dir_x = 2              # Food on right
    else:                                               food_dir_x = 0

    if (snake_head_y > food_y):                         food_dir_y = 1              # Food on top
    elif (snake_head_y < food_y):                       food_dir_y = 2              # Food on bottom
    else:                                               food_dir_y = 0

    if (snake_head_x, snake_head_y-40) in snake_body:   adjoining_body_top = 1      # Body on top
    else:                                               adjoining_body_top = 0

    if (snake_head_x, snake_head_y+40) in snake_body:   adjoining_body_bottom = 1   # Body on bottom
    else:                                               adjoining_body_bottom = 0

    if (snake_head_x-40, snake_head_y) in snake_body:   adjoining_body_left = 1     # Body on left
    else:                                               adjoining_body_left = 0

    if (snake_head_x+40, snake_head_y) in snake_body:   adjoining_body_right = 1    # Body on right
    else:                                               adjoining_body_right = 0
    
    return (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
