'''The mountain car problem, although fairly simple, is commonly applied because it requires a reinforcement learning agent to learn
on two continuous variables: position and velocity. For any given state (position and velocity) of the car, the agent is given the
possibility of driving left, driving right, or not using the engine at all'''
# https://en.wikipedia.org/wiki/Mountain_car_problem
# environment: https://github.com/openai/gym/wiki/MountainCar-v0
# 3 actions: 0:push_left, 1:no_push, 2:push_right
# 2 observations: 0:position ; 1:volecity
# inspired by https://github.com/llSourcell/Q_Learning_Explained

import numpy as np

import gym
from gym import wrappers
from datetime import datetime

# initializations
number_states = 40 # number_of_states
max_iteration = 10000 # max_iteration
initial_learning_rate = 1.0 # initial learning rate
min_learning_rate = 0.005   # minimum learning rate
max_step = 10000 # max_step

# parameters for q learning
epsilon = 0.05
gamma = 1.0

policy_filename = "learned_policy_iter_100000_2019_10_30_15_56.npy"
solution_policy = np.load(policy_filename)

def observation_to_state(environment, observation):
    # map an observation to state
    environment_low = environment.observation_space.low
    environment_high = environment.observation_space.high
    environment_dx = (environment_high - environment_low) / number_states

    # observation[0]:position ;  observation[1]: volecity
    p = int((observation[0] - environment_low[0])/environment_dx[0])
    v = int((observation[1] - environment_low[1])/environment_dx[1])
    # p:position, v:volecity
    return p, v


def episode_simulation(environment, policy=None, render=False):
    observation= environment.reset()
    init_obs = observation
    total_reward = 0
    step_count = 0
    for i in range(max_step):
        if policy is None:
            action = environment.action_space.sample()
        else:
            p,v = observation_to_state(environment, observation)
            action = policy[p][v]
        if render:
            environment.render()
        # proceed environment for each step
        # get observation, reward and done after each step
        observation, reward, done, _ = environment.step(action)
        total_reward += gamma ** step_count * reward
        step_count += 1
        if done:
            break
    total_steps = i
    return init_obs,total_reward,total_steps


if __name__ == '__main__':
    # use gym environment: MountainCar-v0
    # https://github.com/openai/gym/wiki/MountainCar-v0
    environment_name = 'MountainCar-v0'
    environment = gym.make(environment_name)
    environment.seed(0)
    np.random.seed(0)

    reward_list = []
    steps_list = []

    for _ in range(100):
        init_obs,reward,steps = episode_simulation(environment, solution_policy, False)
        steps_list.append(steps)
        reward_list.append(reward)
        print("Inital Observation: "+str(init_obs)+"    Total Reward: "+ str(reward)+"    Total Steps Taken: "+str(steps))
    environment.close()

    print("Final Mean Reward:  "+str(np.mean(np.asarray(reward_list))) + "Final Mean Steps:  "+str(np.mean(np.asarray(steps_list))))
