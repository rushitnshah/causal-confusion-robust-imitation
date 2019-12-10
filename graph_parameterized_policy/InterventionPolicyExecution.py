from pprint import pprint
from time import perf_counter
import pickle
import numpy as np
import gym
import torch
from torch.distributions import Bernoulli
from sklearn.linear_model import Ridge

# initializations
number_states = 40 # number_of_states
max_iteration = 10000 # max_iteration
initial_learning_rate = 1.0 # initial learning rate
min_learning_rate = 0.005   # minimum learning rate
max_step = 10000 # max_step

# parameters for q learning
epsilon = 0.05
gamma = 1.0

"""Intervention using Policy Execution"""

filename = 'mul_lr_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
solution_policy = loaded_model

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


def episode_simulation(environment, data, mask, policy=None, render=False):
    observation= environment.reset()
    init_obs = observation
    total_reward = 0
    step_count = 0
    for i in range(max_step):
        # prepare data tuple for current step
        if len(data)==0:
            prev_action = environment.action_space.sample()
        else:
            prev_action = data[-1][3] #action is the last element in the data row

        p,v = observation_to_state(environment, observation)
        X1 = np.asarray([p,v,prev_action])
        X2 = np.multiply(X1,mask)
        X_G = np.concatenate((X2,mask))

        if policy is None:
            action = environment.action_space.sample()
        else:
            action = solution_policy.predict(X_G.reshape(1,-1))[0]
        if render:
            environment.render()

        data.append([p,v,prev_action,action])
        # proceed environment for each step
        # get observation, reward and done after each step
        observation, reward, done, _ = environment.step(action)
        total_reward += gamma ** step_count * reward
        step_count += 1
        if done:
            break
    total_steps = i
    return init_obs,total_reward,total_steps, data

def sample(weights, temperature):
    return Bernoulli(logits=torch.from_numpy(weights) / temperature).sample().long().numpy()

def linear_regression(masks, rewards, alpha=1.0):
    model = Ridge(alpha).fit(masks, rewards)
    return model.coef_, model.intercept_

if __name__ == '__main__':
    # use gym environment: MountainCar-v0
    # https://github.com/openai/gym/wiki/MountainCar-v0
    environment_name = 'MountainCar-v0'
    environment = gym.make(environment_name)
    environment.seed(0)
    np.random.seed(0)
    # run with render=True for visualization
    reward_list = []
    steps_list = []
    data = []
    weights = np.zeros(3)
    t = 10
    masks = []
    rewards = []
    trace = []
    for it in range(1000):
        start = perf_counter()
        mask = sample(weights, t)
        ep_reward_list = []
        for _ in range(100):
            init_obs,ep_reward,steps,updated_data = episode_simulation(environment, data, mask, solution_policy, True)
            ep_reward_list.append(ep_reward)
            data = updated_data

        reward = np.mean(ep_reward_list)
        masks.append(mask)
        rewards.append(reward)
        weights, _ = linear_regression(masks, rewards, alpha=1.0)

        trace.append(
                {
                    "it": it,
                    "reward": reward,
                    "mask": mask,
                    "weights": weights,
                    "mode": (np.sign(weights).astype(np.int64) + 1) // 2,
                    "time": perf_counter() - start,
                    "past_mean_reward": np.mean(rewards),
                }
            )
        pprint(trace[-1])
    environment.close()
