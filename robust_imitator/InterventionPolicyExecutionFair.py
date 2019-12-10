from pprint import pprint
from time import perf_counter
import pickle
import numpy as np
import pandas as pd
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

filename = 'fair_lr_model.sav'
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
def prepare_data(X1):
    dataX = pd.DataFrame({'Column1': X1[:,0], 'Column2': X1[:,1]})
    dataA = pd.DataFrame({'1': X1[:,2]})

    return dataA.iloc[:,0],dataX
def normalize_data(dataA,dataX):
    ts_X = dataX
    ts_A = dataA
    ts_X = pd.concat([ts_X, ts_A], axis=1)
    for c in list(ts_X.columns):
            if ts_X[c].min() < 0 or ts_X[c].max() > 1:
                if c is 'Column1':
                    ts_X.loc[:,c] = (ts_X[c] - 16.3771) / 9.7967
                elif c is 'Column2':
                    ts_X.loc[:,c] = (ts_X[c] - 21.3542) / 8.3881
    return ts_X
def episode_simulation(environment, data, policy=None, render=False):
    observation= environment.reset()
    init_obs = observation
    total_reward = 0
    step_count = 0
    for i in range(max_step):
        # prepare data tuple for current step
        if len(data)==0:
            prev_action = environment.action_space.sample()
            if prev_action is 1:
                if np.random.sample()>0.5:
                    prev_action = 2
                else:
                    prev_action = 0
        else:
            prev_action = data[-1][3] #action is the last element in the data row

        if prev_action is 2:
            prev_action = 1

        p,v = observation_to_state(environment, observation)
        X1 = np.asarray([p,v,prev_action])
        X1 = np.asarray([X1,X1])
        dataA, dataX = prepare_data(X1)
        ts_X = normalize_data(dataA,dataX)
        testX = np.asarray([ts_X.values[0,0:3],ts_X.values[0,0:3]])
        testA = np.asarray([ts_X.values[0,2],ts_X.values[0,2]])
        if policy is None:
            action = environment.action_space.sample()
        else:
            action = solution_policy.predict(testX,testA)[0]
            if action > 0:
                action = 2.0
        if render:
            environment.render()

        data.append([p,v,prev_action,action])
        # proceed environment for each step
        # get observation, reward and done after each step
        observation, reward, done, _ = environment.step(int(action))
        total_reward += gamma ** step_count * reward
        step_count += 1
        if done:
            break
    total_steps = i
    return init_obs,total_reward,total_steps, data

if __name__ == '__main__':
    # use gym environment: MountainCar-v0
    # https://github.com/openai/gym/wiki/MountainCar-v0
    environment_name = 'MountainCar-v0'
    environment = gym.make(environment_name)
    # environment.seed(0)
    # np.random.seed(0)
    # run with render=True for visualization
    reward_list = []
    steps_list = []
    data = []
    rewards = []
    trace = []
    for it in range(1000):
        start = perf_counter()
        ep_reward_list = []
        for _ in range(1):
            init_obs,ep_reward,steps,updated_data = episode_simulation(environment, data, solution_policy, True)
            ep_reward_list.append(ep_reward)
            data = updated_data

        reward = np.mean(ep_reward_list)
        rewards.append(reward)
        

        trace.append(
                {
                    "it": it,
                    "reward": reward,
                    "time": perf_counter() - start,
                    "past_mean_reward": np.mean(rewards),
                }
            )
        pprint(trace[-1])
        # steps_list.append(steps)
        # reward_list.append(reward)
        # print("Inital Observation: "+str(init_obs)+"    Total Reward: "+ str(reward)+"    Total Steps Taken: "+str(steps))
    environment.close()

    # print("Final Mean Reward:  "+str(np.mean(np.asarray(reward_list))) + "Final Mean Steps:  "+str(np.mean(np.asarray(steps_list))))


    ## Save the data
    # print("Length of data = " + str(len(data)))
    # date_obj = datetime.today()
    # time_str = str(date_obj.year)+"_"+str(date_obj.month)+"_"+str(date_obj.day)+"_"+str(date_obj.hour)+"_"+str(date_obj.minute)
    # filename = "/home/baxter2/Desktop/causal_confusion/custom/Qlearning_MountainCar/expert_data_"+time_str
    # np.save(filename,np.asarray(data))
