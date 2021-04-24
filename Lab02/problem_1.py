## Created by: #Bix Eriksson(940113-3153) & Chieh-Ju Wu (960713-4815)
 ## Reinforcement Learning EL 2805 - Royal Institute of Technology 

import gym
import numpy as np
from ddqn import Agent

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 500
    load_model = True

    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-4, input_dims=[8],
                  batch_size=64, n_actions=4, max_mem_size=100000,
                  eps_min=0.01, eps_dec=1e-3, replace=100)

    if load_model:
        agent.load_models()

    scores, eps_history = [], []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        # [pos_x, pos_y, vel_x, vel_y, angle, angular_vel, left_cnt_point, right_cnt_point]

        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('episode: ', i,
              '\tscore: %.2f' % score,
              '\taverage score: %.2f' % avg_score,
              '\tepsilon: %.2f' % agent.epsilon)

        # save model every 10 episodes
        if i > 0 and i % 10 == 0:
            agent.save_models()

        if avg_score >= 150:
            agent.save_models()
            break
    # env.close()
