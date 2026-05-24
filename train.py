import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env


def relative_improvement(best_value, current_value, eps=1e-9):
    if not np.isfinite(best_value):
        return 1.0
    return (best_value - current_value) / max(abs(best_value), eps)

def setup_seed(seed):
    torch.manual_seed(seed)  # 设置CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    np.random.seed(seed)  # 设置numpy的随机种子
    random.seed(seed)  # 设置Python的随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次运行结果相同

def main():
    # PyTorch initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor') # Use GPU
    else:
        torch.set_default_tensor_type('torch.FloatTensor')  # Use CPU
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)  # 设置打印选项

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]  # 环境参数
    model_paras = load_dict["model_paras"]  # 模型参数
    train_paras = load_dict["train_paras"]  # 训练参数
    env_paras["device"] = device 
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)  # 验证环境参数
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]  # 验证批次大小
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2  # Actor输入维度
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]  # Critic输入维度

    reward_weights = env_paras.get("reward_weights", {})
    w_makespan = float(reward_weights.get("makespan", 1.0))
    w_energy = float(reward_weights.get("energy", 1.0))
    w_penalty = float(reward_weights.get("penalty", 1.0))

    num_jobs = env_paras["num_jobs"]  # 作业数量
    num_mas = env_paras["num_mas"]  # 机器数量
    opes_per_job_min = int(num_mas * 0.8)  # 每个作业的最小操作数
    opes_per_job_max = int(num_mas * 1.2)  # 每个作业的最大操作数

    memories = PPO_model.Memory()  # 
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])  # Initialize PPO model
    env_valid = get_validate_env(env_valid_paras)  # Create an environment for validation
    maxlen = 1  # Save the best model
    best_models = deque()
    best_reward_makespan = float('inf')
    best_reward_energy = float('inf')
    best_reward_penalty = float('inf')

    # Use visdom to visualize the training process
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])

    os.environ['TZ'] = 'Asia/Shanghai'  # Set timezone to Shanghai
    time.tzset()  # Apply timezone change

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))  # Current time string
    save_path = './save/train_{0}'.format(str_time)  # Save path for training results
    os.makedirs(save_path)
    # Training curve storage path (average of validation set)
    writer_ave = pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time))
    # Training curve storage path (value of each validating instance)
    writer_100 = pd.ExcelWriter('{0}/training_100_{1}.xlsx'.format(save_path, str_time))
    makespan_results = []
    energy_results = []
    penalty_results = []
    makespan_results_100 = []
    energy_results_100 = []
    penalty_results_100 = []
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])  # Data for training iterations
    data_file.to_excel(writer_ave, sheet_name='makespan', index=False)
    data_file.to_excel(writer_ave, sheet_name='energy', index=False)
    data_file.to_excel(writer_ave, sheet_name='penalty', index=False)
    writer_ave.save()
    writer_ave.close()
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])  # Data for training iterations (100 instances)
    data_file.to_excel(writer_100, sheet_name='makespan', index=False)
    data_file.to_excel(writer_100, sheet_name='energy', index=False)
    data_file.to_excel(writer_100, sheet_name='penalty', index=False)
    writer_100.save()
    writer_100.close()

    # Start training iteration
    start_time = time.time()
    env = None
    for i in range(1, train_paras["max_iterations"]+1):
        # Replace training instances every x iteration (x = 20 in paper)
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # \mathcal{B} instances use consistent operations to speed up training
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        # Get state and completion signal, and reward of each environment
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time()-last_time)

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()
            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                    win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated

        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            makespan, makespan_100, energy, energy_100, penalty, penalty_100 = validate(env_valid_paras, env_valid, model.policy_old)
            makespan_results.append(makespan.item())
            makespan_results_100.append(makespan_100)
            energy_results.append(energy.item())
            penalty_results.append(penalty.item())
            energy_results_100.append(energy_100)
            penalty_results_100.append(penalty_100)

            # Save the best model
            makespan_now = makespan.item()
            energy_now = energy.item()
            penalty_now = penalty.item()

            temp_reward = (
                w_makespan * relative_improvement(best_reward_makespan, makespan_now)
                + w_energy * relative_improvement(best_reward_energy, energy_now)
                + w_penalty * relative_improvement(best_reward_penalty, penalty_now)
            ) / max(w_makespan + w_energy + w_penalty, 1e-9)
            print("temp_reward: ", '%.6f' % temp_reward)
            print()
            if temp_reward > 0:
                best_reward_makespan = makespan_now
                best_reward_energy = energy_now
                best_reward_penalty = penalty_now
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([makespan.item()]),
                    win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))
                viz.line(
                    X=np.array([i]), Y=np.array([energy.item()]),
                    win='window{}'.format(3), update='append', opts=dict(title='energy of valid'))
                viz.line(
                    X=np.array([i]), Y=np.array([penalty.item()]),
                    win='window{}'.format(4), update='append', opts=dict(title='penalty of valid'))

    # Save the data of training curve to files
    data = pd.DataFrame(np.array(makespan_results).transpose(), columns=["res"])
    data.to_excel(writer_ave, sheet_name='makespan', index=False, startcol=1)
    data = pd.DataFrame(np.array(energy_results).transpose(), columns=["res"])
    data.to_excel(writer_ave, sheet_name='energy', index=False, startcol=1)
    data = pd.DataFrame(np.array(penalty_results).transpose(), columns=["res"])
    data.to_excel(writer_ave, sheet_name='penalty', index=False, startcol=1)
    writer_ave.save()
    writer_ave.close()

    column = [i_col for i_col in range(100)]
    data = pd.DataFrame(np.array(torch.stack(makespan_results_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100, sheet_name='makespan', index=False, startcol=1)
    data = pd.DataFrame(np.array(torch.stack(energy_results_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100, sheet_name='energy', index=False, startcol=1)
    data = pd.DataFrame(np.array(torch.stack(penalty_results_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100, sheet_name='penalty', index=False, startcol=1)
    writer_100.save()
    writer_100.close()

    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()

    