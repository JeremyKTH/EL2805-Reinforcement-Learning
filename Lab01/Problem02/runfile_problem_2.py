##Created by:
#Bix Eriksson(940113-3153) & Chieh-Ju Wu (960713-4815)
#Reinforcement Learning EL 2805 - Royal Institute of Technology
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import problem_2 as p2

##Generate a gameMap, 1 = Bank Position
gameMap = np.array([[1, 0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 1]])


env =p2.Bank(gameMap, start_robber_pos=(0,0), start_police_pos=(1,2))
##Generate a policy
alpha = 0.8
policy, V = env.value_iteration(alpha)

##Plot value function as a function of alpha
#env.value_function_of_alpha(start_robber_pos=(0,0), start_police_pos=(1,2))

##Illustrates the optimal policy
env.animate_solution(gameMap, policy, (0,0), (1,2))
