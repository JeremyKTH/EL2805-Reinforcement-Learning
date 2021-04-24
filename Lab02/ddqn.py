## Created by: #Bix Eriksson(940113-3153) & Chieh-Ju Wu (960713-4815)
 ## Reinforcement Learning EL 2805 - Royal Institute of Technology 

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ReplayBuffer(object):
    def __init__(self, max_mem_size, input_dims, n_actions):
        self.mem_size = max_mem_size
        self.mem_cntr = 0 # counter for sizing replay buffer
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        # when full, overwrite from the beginnning
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # before filling up the max memory buffer (np.array),
        # use mem_cntr to avoid empty entries
        max_mem = min(self.mem_cntr, self.mem_size)
        # replace = False, avoid sampling the same sample again
        batch = np.random.choice(max_mem, batch_size, replace=False)

        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        return state_batch, action_batch, reward_batch, new_state_batch, terminal_batch

class DuelDeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DuelDeepQNetwork, self).__init__()

        # Deep Nerual Network - fully connected
        # 2 hidden layers, 128 neurons per layer
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.V = nn.Linear(128, 1) #value function
        self.A = nn.Linear(128, n_actions) #advantage function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        V = self.V(l2) # value function
        A = self.A(l2) # advantage function

        return V, A

    def save_checkpoint(self):
        print('...saving checkpoint...')
        PATH = "/Users/jwgsolitude/desktop/neural-network-1.pth"
        T.save(self.state_dict(), PATH)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        PATH = "/Users/jwgsolitude/desktop/neural-network-1.pth"
        self.load_state_dict(T.load(PATH))

class Agent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                    max_mem_size, eps_min, eps_dec, replace):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.update_every_C_steps = replace # update Q target network every C steps
        self.tar_cntr = 0 # counter for replacing Q target network

        self.memory = ReplayBuffer(max_mem_size, input_dims, n_actions)
        self.q_eval = DuelDeepQNetwork(lr=lr, input_dims=input_dims, n_actions=n_actions)
        self.q_next = DuelDeepQNetwork(lr=lr, input_dims=input_dims, n_actions=n_actions)

    # store transition in agent memory
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    # epsilon-greedy policy
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = observation[np.newaxis, :]
            state = T.tensor(observation).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item() #get integer
        else:
            action = np.random.choice(self.action_space)

        return action

    def update_target_network(self):
        # update Q target network every C steps
        if self.tar_cntr % self.update_every_C_steps == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

    def learn(self):
        # start learning when batch size is filled
        if self.memory.mem_cntr < self.batch_size:
            return

        # PyTorch requires resetting the gradient of optimizer
        self.q_eval.optimizer.zero_grad()

        self.update_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state).to(self.q_eval.device)
        new_state = T.tensor(new_state).to(self.q_eval.device)
        action = T.tensor(action).to(self.q_eval.device)
        reward = T.tensor(reward).to(self.q_eval.device)
        done = T.tensor(done).to(self.q_eval.device)

        V_s, A_s = self.q_eval.forward(state) #Value & Advantage at state s
        V_s_, A_s_ = self.q_next.forward(new_state)

        # predicted
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1, action.unsqueeze(-1)).squeeze(-1)
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_target = reward + self.gamma* T.max(q_next, dim=1)[0].detach()
        q_target[done] = 0.0

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.tar_cntr += 1

        self.decrement_epsilon()


    def save_models(self):
        # self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        # self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
