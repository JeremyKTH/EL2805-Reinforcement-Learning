## Created by: #Bix Eriksson(940113-3153) & Chieh-Ju Wu (960713-4815)
 ## Reinforcement Learning EL 2805 - Royal Institute of Technology 

import numpy as np
import gym
import torch
from tqdm import trange
from ddqn import DuelDeepQNetwork

# Plotting
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


# load model
FILE = "/Users/jwgsolitude/desktop/neural-network-1.pth"
model = DuelDeepQNetwork(lr=5e-4, input_dims=[8], n_actions=4)
model.load_state_dict(torch.load(FILE))

# setup environment
env = gym.make('LunarLander-v2')
state = env.reset()

# setup limitations for the space ship
# state = [pos_x, pos_y, vel_x, vel_y, angle, angular_vel, left_cnt_points, right_cnt_points]
height = np.linspace(0, 1.5, num=10)
angle = np.linspace(-np.pi, np.pi, num=10)

#actions_list = np.zeros((15, 15))
q_values_list = np.zeros((10, 10))

for i in range(10):
    for j in range(10):
        state = [0., height[i], 0., 0., angle[j], 0., 0., 0.]

        q_values = model(torch.tensor([state]))
        #_, action = torch.max(q_values[1], axis=1)
        # print(q_values)
        # save q_value into 2d nparray
        q_value = q_values[0].squeeze().item()
        q_values_list[j, i] = q_value

        # save action into 2d nparray
        #actions_list[i, j] = action

        j += 1
    i += 1

## Plot
plt.show()

fig = plt.figure()

height, angle = np.meshgrid(height, angle)

ax = plt.axes(projection='3d')
ax.set_xlabel('Height')
ax.set_ylabel('Angle')
ax.set_zlabel('Q values')
ax.set_title('Q(s, a)')

ax.plot_surface(height, angle, q_values_list, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

plt.show()
