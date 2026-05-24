import gym
import env
import PPO_model
import torch
import time
import os
import copy

def get_validate_env(env_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    file_path = "./data_dev1/{0}{1}/raw/".format(env_paras["num_jobs"], str.zfill(str(env_paras["num_mas"]),2))
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=env_paras, data_source='file')
    return env

def validate(env_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    makespan = copy.deepcopy(env.makespan_batch.mean())  # 记录平均makespan
    makespan_batch = copy.deepcopy(env.makespan_batch)  # 记录每个实例的makespan
    energy = copy.deepcopy(env.energy_batch.mean())  # 记录平均energy
    energy_batch = copy.deepcopy(env.energy_batch)  # 记录每个实例的energy
    penalty = copy.deepcopy(env.penalty_batch.mean())  # 记录平均penalty
    penalty_batch = copy.deepcopy(env.penalty_batch)  # 记录每个实例的penalty
    env.reset()
    print('validating time: ', time.time() - start)
    print('makespan: ', makespan.item(), 'energy: ', energy.item(), 'penalty: ', penalty.item(), '\n')
    return makespan, makespan_batch, energy, energy_batch, penalty, penalty_batch