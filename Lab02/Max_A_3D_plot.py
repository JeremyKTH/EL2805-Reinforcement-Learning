## Created by: #Bix Eriksson(940113-3153) & Chieh-Ju Wu (960713-4815)
 ## Reinforcement Learning EL 2805 - Royal Institute of Technology 

import numpy as np
import gym
import torch
from tqdm import trange
from ddqn import DuelDeepQNetwork
import matplotlib.cm as cm

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


height = np.linspace(0, 1.5, num=15)
angle = np.linspace(-np.pi, np.pi, num=15)

actions_list = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        state = [0., height[i], 0., 0., angle[j], 0., 0., 0.]

        q_values = model(torch.tensor([state]))
        _, action = torch.max(q_values[1], axis=1)

        actions_list[j, i] = action

        j += 1
    i += 1

# actions_list = np.array(actions_list)

fig = plt.figure()

height, angle = np.meshgrid(height, angle)

ax = plt.axes(projection='3d')
ax.set_xlabel('Height')
ax.set_ylabel('Angle')
ax.set_zlabel('action')
ax.set_title('Optimal Action')
# ax.zaxis.set_rotate_label(False)
# ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_zticks([0, 1, 2, 3])
ax.set_zlim(0, 3)

"""surf = ax.plot_surface(height, angle, actions_list, rstride=1, cstride=1, cmap='viridis', edgecolor='none');"""

ax.scatter(height, angle, actions_list, s=20)


plt.show()
