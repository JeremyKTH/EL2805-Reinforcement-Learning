## Created by: #Bix Eriksson(940113-3153) & Chieh-Ju Wu (960713-4815)
 ## Reinforcement Learning EL 2805 - Royal Institute of Technology 

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random


# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Maze:

    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    GOAL_REWARD = 1
    DEAD_REWARD = -100
    TERMINAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100

    def __init__(self, maze, weights=None, random_rewards=False, minotaur_cant_stay=True):
        self.maze = maze
        self.minotaur_cant_stay = minotaur_cant_stay
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards(weights=weights,
                                      random_rewards=random_rewards)
    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        # S = {Px, Py, Mx, My}
        # Player position
        for px in range(self.maze.shape[0]):
            for py in range(self.maze.shape[1]):
                # Minotaur position
                for mx in range(self.maze.shape[0]):
                    for my in range(self.maze.shape[1]):
                        # All combinations of player and minotaur
                        # inside the maze and player not in wall
                        if self.maze[px, py] != 1:
                            states[s] = (px, py, mx, my)
                            map[(px, py, mx, my)] = s
                            s += 1
        # 2 Terminal States
        states[s] = 'WIN'
        map['WIN'] = s
        s += 1
        states[s] = 'DEAD'
        map['DEAD'] = s
        return states, map

    def __move(self, state, action, prob=False):
        """ Player makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the player stays in place.
            Simultaneously the minotaur makes the move

            for prob:
                returns the length of valid minotaur positions to get its transition probability

            :return tuple next_state:
                (Px,Py,Mx,My) on the maze that player and minotaur transitions to.
        """
        # --- player ---
        # Check if reached Terminal states: Win/Dead
        if self.states[state] == 'WIN' or self.is_win(state):
            return self.map['WIN']
        if self.states[state] == 'DEAD' or self.is_dead(state):
            return self.map['DEAD']

        # Compute the future position given current (state, action)
        row_p = self.states[state][0] + self.actions[action][0]
        col_p = self.states[state][1] + self.actions[action][1]

        # Check if player hits the wall or go out of boundary
        valid_move_p = (row_p >= 0) and (row_p < self.maze.shape[0]) and \
            (col_p >= 0) and (col_p < self.maze.shape[1]) and not \
            self.maze[row_p, col_p] == 1

        # If invalid move --> STAY
        if not valid_move_p:
            row_p = self.states[state][0]
            col_p = self.states[state][1]

        # --- minotaur ---
        minotaur_actions = list(self.actions.keys())
        if self.minotaur_cant_stay:
            minotaur_actions.remove(self.STAY)


        # Find minotaur's next available position
        m_next_states = []
        for action in minotaur_actions:
            row_m = self.states[state][2] + self.actions[action][0]
            col_m = self.states[state][3] + self.actions[action][1]
            valid_move_m = (row_m >= 0) and (row_m < self.maze.shape[0]) and \
                (col_m >= 0) and (col_m < self.maze.shape[1])
            if valid_move_m:
                m_next_states.append([row_m, col_m])

        if prob:
            return row_p, col_p, m_next_states
        else:
            # minotaur random walk
            minotaur_pos = random.choice(m_next_states)
            return self.map[(row_p, col_p, minotaur_pos[0], minotaur_pos[1])]

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # P(S' = 'DEAD' | S = 'DEAD', A = (.)) = 1
                # P(S' = 'WIN' | S = 'WIN', A = (.)) = 1
                if self.states[s] == 'WIN' or self.states[s] == 'DEAD':
                    next_s = s
                    transition_probabilities[next_s, s, a] = 1

                # P(S' = 'DEAD' | S = (Pxy == Mxy), A = (.)) = 1
                elif self.is_dead(s):
                    next_s = self.map['DEAD']
                    transition_probabilities[next_s, s, a] = 1

                # P(S' = 'WIN' | S = (Pxy == Gxy), A = (.)) = 1
                elif self.is_win(s):
                    next_s = self.map['WIN']
                    transition_probabilities[next_s, s, a] = 1

                else:
                    row, col, m_next_states = self.__move(s, a, prob=True)
                    for minotaur_pos in m_next_states:
                        next_s = self.map[(row, col, minotaur_pos[0], minotaur_pos[1])]
                        transition_probabilities[next_s, s, a] = 1 / len(m_next_states)

        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):
        rewards = np.zeros((self.n_states, self.n_actions))
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    # --- terminal state 'WIN' or 'DEAD' ---
                    if self.states[s] == 'WIN' or self.states[s] == 'DEAD':
                        next_s = s
                        rewards[s,a] = self.TERMINAL_REWARD
                        continue
                    # --- non-terminal state ---
                    next_s = self.__move(s, a)
                    # Reward for hitting a wall or out of boundary
                    if s == next_s and a != self.STAY:
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    # Reward for going in the terminal state - win
                    elif s == next_s and self.is_win(s):
                        rewards[s, a] = self.TERMINAL_REWARD
                    # Reward for going in the terminal state - dead
                    elif s == next_s and self.is_dead(s):
                        rewards[s, a] = self.TERMINAL_REWARD
                    # Reward for reaching the goal
                    elif self.is_win(s):
                        rewards[s, a] = self.GOAL_REWARD
                    # Reward for being dead
                    elif self.is_dead(s):
                        rewards[s, a] = self.DEAD_REWARD
                    # Reward for a normal step
                    else:
                        rewards[s, a] = self.STEP_REWARD

        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s, a)
                    i, j = self.states[next_s]
                    # Simply put the reward as the weights o the next state.
                    rewards[s, a] = weights[i][j]

        return rewards

    def is_win(self, s):
        # S = {Px, Py, Mx, My}
        return self.maze[self.states[s][0:2]] == 2 and not self.is_dead(s)

    def is_dead(self, s):
        return self.states[s][0:2] == self.states[s][2:4]

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s, t])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s

        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T-1, -1, -1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Update the color at each frame
    for i in range(len(path)):

        ax.set_title(f'\t \t \t \t  Policy simulation \t \t \t t {i} T {len(path)-1}'.expandtabs())
        # First clear the prev illustration, if path[i] is same as path[i-1] then it is already changed!
        # Illustration of current status
        if i > 0:
        # if path[i][0:2] != path[i-1][0:2]:
            grid.get_celld()[(path[i-1][0:2])
                            ].set_facecolor(col_map[maze[path[i-1][0:2]]])
            grid.get_celld()[(path[i-1][0:2])].get_text().set_text('')
        # if path[i][2:4] != path[i-1][2:4]:
            grid.get_celld()[(path[i-1][2:4])
                            ].set_facecolor(col_map[maze[path[i-1][2:4]]])
            grid.get_celld()[(path[i-1][2:4])].get_text().set_text('')


        # Agent illustration
        grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0:2])].get_text().set_text('Player')
        grid.get_celld()[(path[i][2:4])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path[i][2:4])].get_text().set_text('Minotaur')

        # Position is the same and it is DEAD!
        if path[i][0:2] == path[i][2:4]:
            grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i][0:2])].get_text().set_text('DEAD')
            break # Since nothing changes
        # Position is the same and it is WIN!
        elif maze[path[i][0:2]] == 2:
            grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_GREEN)
            grid.get_celld()[(path[i][0:2])].get_text().set_text('WIN')
            break # Since nothing changes

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
