# EL2805 Reinforcement Learning
This is a repo for the course assignments of [DD2424 Deep Learning in Data Science](https://www.kth.se/student/kurser/kurs/EL2805?l=en) at KTH 2020. The code in this repo is mainly done in Python, GoogleColab and Jupyter Notebook.

## Table of contents

<!--ts-->
   * [Lab01](#Lab01)
      * [Problem01](#Problem01)
      * [Problem01](#Problem01)
   * [Lab02](#Lab02)
<!--te-->


## Lab01

### Problem01

The Maze and the Random Minotaur

### Problem02

Robbing Banks with value and policy iteration

## Lab02
Lunar Lander with Dueling Deep Q-Learning and Experience Replay

### Introduction
This project implements the [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) from OpenAI's Gym with Pytorch.<br/> 
The goal is to manoeuvre the space ship so that it lands between the two flags. The landing pad is always at coordinates (0, 0). The coordinates are the first two numbers in the state vector. Reward for moving from the top of the screen to the landing pad and zero speed is about 100 ∼ 140 points. If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or comes to rest, receiving an additional −100 or +100 points. Each leg with ground contact is +10 points. Firing the main engine is −0.3 points each frame. Firing the side engine is −0.03 points each frame.

### Results
![gif](https://github.com/JeremyKTH/EL2805-Reinforcement-Learning/blob/main/Lab02/lunarlander.gif)


<!-- CONTACT -->
## Contact

Chieh-Ju Wu (Jeremy) - jeremy.cjwukth@gmail.com
