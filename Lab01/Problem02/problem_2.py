##Created by:
#Bix Eriksson(940113-3153) & Chieh-Ju Wu (960713-4815)
#Reinforcement Learning EL 2805 - Royal Institute of Technology
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import random
import copy
import time

class Bank:

    ##ACTION DEFINITION
    stay = 0
    up = 1
    right = 2
    down = 3
    left = 4

    def __init__(self, gameMap, start_robber_pos, start_police_pos):
        self.gameMap = gameMap
        self.states, self.map = self.__states()
        self.actions = self.__actions()
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.rewards = self.__rewards()
        self.p_matrix = self.__create_p_matrix(start_robber_pos, start_police_pos)
    def __states(self):
        #Returns a dictionary of states and map
        #Each state is represented by every combination of (Ry, Rx, Py, Px)
        #Map holds state nr for a given position
        states = dict()
        map = dict()
        s = 0
        ## (i,j) robber position, (k, t) police position
        for i in range(self.gameMap.shape[0]):
            for j in range(self.gameMap.shape[1]):

                for k in range(self.gameMap.shape[0]):
                    for t in range(self.gameMap.shape[1]):
                        states[s] = (i, j, k, t)
                        map[(i, j, k, t)] = s
                        s += 1
        return states, map

    def __actions(self):
        ##Returns action space
        actions = dict()
        actions[self.stay] = (0, 0)
        actions[self.up] = (-1, 0)
        actions[self.right] = (0, 1)
        actions[self.down] = (1, 0)
        actions[self.left] = (0, -1)
        return actions

    def __robber_move(self, y, x, a):
        ##Input: Robber Position and an action
        ##Return new position if valid move(Not out of boundaries) and old position if in-valid move
        new_y = y + self.actions[a][0]; new_x = x + self.actions[a][1]
        if self.__is_valid_move(new_y, new_x):
            return (new_y, new_x)
        else:
            return (y, x)

    def __police_move(self, Yr, Xr, Yp, Xp, find_possible_moves=None):
        ##Input: Robber position: Yr, Xr and Police position Yp, Xp
        ##
        ##If find possible_move = True:
        ##Returns a list of possible police moves given robber and police position
        ##
        ##If find_possible_move = None:
        ##Returns a new Police position given robber and police position

        possible_moves=list()
        stay = 0
        up = 1
        right = 2
        down = 3
        left = 4
        moves = self.actions.copy()
        ##Removes from moves that are not possible:

        ##Same Column and robber is above police
        if ( (Xr == Xp) and (Yr < Yp) ):
            del moves[down]; del moves[stay]
        ##Same Column and robber is below police
        elif ( (Xr == Xp) and (Yr > Yp) ):
            del moves[up]; moves[stay]
        ##Same Row and robber is to the right of police
        elif( (Yr == Yp) and (Xr > Xp) ):
            del moves[left]; del moves[stay]
        ##Same Row and robber is to the left of police
        elif( (Yr == Yp) and (Xr < Xp) ):
            del moves[right]; del moves[stay]
        ##Robber is up and to the right of police
        elif( (Xr > Xp) and (Yr < Yp) ):
            del moves[left]; del moves[stay]; del moves[down]
        ##Robber is up and to the left of police
        elif( (Xr < Xp) and (Yr < Yp) ):
            del moves[right]; del moves[stay]; del moves[down]
        ##Robber is down and to the right of police
        elif( (Xr > Xp) and (Yr > Yp) ):
            del moves[left]; del moves[stay]; del moves[up]
        ##Robber is down and to the left of police
        elif( (Xr < Xp) and (Yr > Yp) ):
            del moves[right]; del moves[stay]; del moves[up]

        ##If True, is used to update probability matrix in function create_p_matrix
        if (find_possible_moves == True):
            for i in moves:
                y = Yp + moves[i][0]
                x = Xp + moves[i][1]
                if self.__is_valid_move(y, x):
                    possible_moves.append( (Yp + self.actions[i][0], Xp+self.actions[i][1] ))
                else:
                    possible_moves.append( (Yp + self.actions[stay][0], Xp+self.actions[stay][1]) )
            return possible_moves
        move = moves[random.choice( list(moves) )]
        ##Calculate new pos
        new_Yp = Yp + move[0]; new_Xp = Xp + move[1]
        is_Valid = self.__is_valid_move(new_Yp, new_Xp)
        if (is_Valid is True):
            return (new_Yp, new_Xp)
        if (is_Valid is False):
            return (Yp + self.actions[stay][0], Xp + self.actions[stay][1])

    def __is_valid_move(self, y, x):
        ##Returns True if valid move, False if invalid
        if  ( (x < 0) or (y < 0) or (y >= self.gameMap.shape[0]) or (x >= self.gameMap.shape[1]) ):
            return False
        else:
            return True

    def __create_p_matrix(self, start_robber_pos, start_police_pos):
        ##Input: Initial state coordinates
        ##Return probability transition matrix
        p = np.zeros((self.n_states, self.n_states, self.n_actions))
        for s in range(self.n_states):
            ##If caught probability of going to initial state = 1 no matter action
            if(self.states[s][0] == self.states[s][2] and self.states[s][1] == self.states[s][3]):
                for a in range(self.n_actions):
                    p[self.map[start_robber_pos + start_police_pos],s, a] = 1
            ##If not caught: Defines probability of going from one state to another
            else:
                for a in range(self.n_actions):
                    next_robber_pos = self.__robber_move(self.states[s][0], self.states[s][1], a)
                    next_possible_police_pos = self.__police_move(self.states[s][0], self.states[s][1], self.states[s][2], self.states[s][3], find_possible_moves=True )
                    for i in range(len(next_possible_police_pos)):
                        p[self.map[(next_robber_pos[0], next_robber_pos[1], next_possible_police_pos[i][0], next_possible_police_pos[i][1])], s, a] = 1/len(next_possible_police_pos)
        return p

    def __rewards(self):
        ##Returns reward vector for each state
        rewards = np.zeros(self.n_states)
        for s in range(self.n_states):
            if (self.states[s][0] == self.states[s][2] and self.states[s][1] == self.states[s][3]):
                rewards[s] = -50
            elif (self.gameMap[self.states[s][0], self.states[s][1]] == 1):
                rewards[s] = 10
        return rewards

    def value_iteration(self, alpha):
        ##Input: discount factor alpha
        V = np.zeros(self.n_states)
        r = self.rewards
        P = self.p_matrix
        policy = np.zeros(self.n_states)
        V_temp = np.ones(self.n_states)
        n = 0
        tol = 0.1
        ##Value Iteration
        while np.linalg.norm(V - V_temp) >= tol and n < 200:
            V=np.copy(V_temp)
            n += 1
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    action_value = np.dot(P[:,s, a], r) + alpha*np.dot(P[:,s,a], V)
                    if (a == 0):
                        action_value_max = action_value
                    else:
                        if(action_value > action_value_max):
                            action_value_max=action_value
                V_temp[s] = action_value_max
        ##Generate Optimal Policy
        for s in range(self.n_states):
            for a in range(self.n_actions):
                action_value= np.dot(P[:,s, a], r) + alpha*np.dot(P[:,s,a], V)
                ##Stores the larges action value and its corresponding policy
                if (a==0):
                    action_value_max=action_value
                    bestPolicy=a
                else:
                    if (action_value > action_value_max):
                        action_value_max=action_value
                        bestPolicy=a
            ##Stores optimal policy for each state
            policy[s]=bestPolicy

        return policy, V

    def value_function_of_alpha(self, start_robber_pos, start_police_pos):
        ##Is used to plot value function at initial state as a function of discount factor alpha
        ##Input: Initial state position (Ry, Rx), (Py, Px)
        initial_state=self.map[start_robber_pos+start_police_pos]
        alpha = (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
        x=alpha
        value = list()
        for i in alpha:
            policy, V = self.value_iteration(i)
            value.append(V[initial_state])

        y=value
        plt.plot(x, y)
        plt.xlabel('Alpha')
        plt.ylabel('Value function')
        plt.show()

    def animate_solution(self, maze, policy, start_robber_pos, start_police_pos):
        robber_pos = start_robber_pos
        police_pos = start_police_pos

        LIGHT_RED    = '#FFC4CC';
        LIGHT_GREEN  = '#95FD99';
        BLACK        = '#000000';
        WHITE        = '#FFFFFF';
        LIGHT_PURPLE = '#E8D0FF';
        LIGHT_ORANGE = '#FAE0C3';
        # Map a color to each cell in the maze
        col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

        # Size of the maze
        rows,cols = maze.shape;

        # Create figure of the size of the maze
        fig = plt.figure(1, figsize=(cols,rows));

        # Remove the axis ticks and add title title
        ax = plt.gca();
        ax.set_title('Policy simulation');
        ax.set_xticks([]);
        ax.set_yticks([]);

        # Give a color to each cell
        colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

        # Create figure of the size of the maze
        fig = plt.figure(1, figsize=(cols,rows))

        # Create a table to color
        grid = plt.table(cellText=None,
                         cellColours=colored_maze,
                         cellLoc='center',
                         loc=(0,0),
                         edges='closed');

        # Modify the hight and width of the cells in the table
        tc = grid.properties()['children']
        for cell in tc:
            cell.set_height(1.0/rows);
            cell.set_width(1.0/cols);

        ##Draw start position
        grid.get_celld()[(robber_pos)].set_facecolor(LIGHT_GREEN)
        grid.get_celld()[(robber_pos)].get_text().set_text('Robber')
        grid.get_celld()[(police_pos)].set_facecolor(LIGHT_RED)
        grid.get_celld()[(police_pos)].get_text().set_text('Police')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(4)

        average_reward=0
        n_steps=0
        t = 1
        # Update the color at each frame
        while t == 1:

            current_state=self.map[robber_pos+police_pos]
            ##Old Position
            old_robber_pos = robber_pos
            old_police_pos = police_pos
            ##New Position
            police_pos = self.__police_move(robber_pos[0], robber_pos[1], police_pos[0], police_pos[1])
            robber_pos=self.__robber_move(robber_pos[0], robber_pos[1], policy[current_state])

            ##Delete old drawing
            if(old_robber_pos != robber_pos):
                if(maze[old_robber_pos[0], old_robber_pos[1]] == 1):
                    grid.get_celld()[(old_robber_pos)].set_facecolor(BLACK)
                    grid.get_celld()[(old_robber_pos)].get_text().set_text('')
                else:
                    grid.get_celld()[(old_robber_pos)].set_facecolor(WHITE)
                    grid.get_celld()[(old_robber_pos)].get_text().set_text('')
            if(old_police_pos != police_pos):
                if(maze[old_police_pos[0], old_police_pos[1]] == 1):
                    grid.get_celld()[(old_police_pos)].set_facecolor(BLACK)
                    grid.get_celld()[(old_police_pos)].get_text().set_text('Bank')
                else:
                    grid.get_celld()[(old_police_pos)].set_facecolor(WHITE)
                    grid.get_celld()[(old_police_pos)].get_text().set_text('')

            ##Draw New Postion
            grid.get_celld()[(robber_pos)].set_facecolor(LIGHT_GREEN)
            grid.get_celld()[(robber_pos)].get_text().set_text('Robber')
            grid.get_celld()[(police_pos)].set_facecolor(LIGHT_RED)
            grid.get_celld()[(police_pos)].get_text().set_text('Police')

            display.display(fig)
            display.clear_output(wait=True)
            time.sleep(1)

            ##Reinitilize map if caught
            if (robber_pos == police_pos):
                old_robber_pos = robber_pos
                old_police_pos = police_pos
                robber_pos = start_robber_pos
                police_pos = start_police_pos
                n_steps +=1

                if(old_robber_pos != robber_pos):
                    if(maze[old_robber_pos[0], old_robber_pos[1]] == 1):
                        grid.get_celld()[(old_robber_pos)].set_facecolor(BLACK)
                        grid.get_celld()[(old_robber_pos)].get_text().set_text('')
                    else:
                        grid.get_celld()[(old_robber_pos)].set_facecolor(WHITE)
                        grid.get_celld()[(old_robber_pos)].get_text().set_text('')
                if(old_police_pos != police_pos):
                    if(maze[old_police_pos[0], old_police_pos[1]] == 1):
                        grid.get_celld()[(old_police_pos)].set_facecolor(BLACK)
                        grid.get_celld()[(old_police_pos)].get_text().set_text('Bank')
                    else:
                        grid.get_celld()[(old_police_pos)].set_facecolor(WHITE)
                        grid.get_celld()[(old_police_pos)].get_text().set_text('')

                grid.get_celld()[(robber_pos)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(robber_pos)].get_text().set_text('Robber')
                grid.get_celld()[(police_pos)].set_facecolor(LIGHT_RED)
                grid.get_celld()[(police_pos)].get_text().set_text('Police')

                display.display(fig)
                display.clear_output(wait=True)
                time.sleep(1)
