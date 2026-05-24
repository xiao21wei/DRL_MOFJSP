import copy
import json
import os
import random
import time as time

import gym
import pandas as pd
import torch
import numpy as np

import pynvml
import PPO_model
from env.load_data import nums_detec

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    pynvml.nvmlInit()  # Initialize NVML for GPU memory management
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get the first GPU handle， 获取GPU句柄
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"]
    reward_weights = env_paras.get("reward_weights", {})
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = test_paras["num_ins"]
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    data_path = "./data_test1/{0}/raw/".format(test_paras["data_path"])
    test_files = os.listdir(data_path)
    test_files.sort(key=lambda x: x[:-4])
    test_files = test_files[:num_ins]
    mod_files = os.listdir('./model/')[:]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras)
    rules = test_paras["rules"]
    envs = []  # Store multiple environments

    # Detect and add models to "rules"
    if "DRL" in rules:
        for root, ds, fs in os.walk('./model/'):
            for f in fs:
                if f.endswith('.pt'):
                    rules.append(f)
    if len(rules) != 1:
        if "DRL" in rules:
            rules.remove("DRL")

    os.environ['TZ'] = 'Asia/Shanghai'  # Set timezone to Shanghai
    time.tzset()  # Apply timezone change

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/test_{0}'.format(str_time)
    os.makedirs(save_path)
    writer = pd.ExcelWriter('{0}/test_{1}.xlsx'.format(save_path, str_time))  # Makespan data storage path
    file_name = [test_files[i] for i in range(num_ins)]
    data_file = pd.DataFrame(file_name, columns=["file_name"])
    data_file.to_excel(writer, sheet_name='makespan', index=False)
    data_file.to_excel(writer, sheet_name='energy', index=False)
    data_file.to_excel(writer, sheet_name='penalty', index=False)
    data_file.to_excel(writer, sheet_name='time', index=False)
    writer.save()
    writer.close()

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_rules in range(len(rules)):
        rule = rules[i_rules]
        # Load trained model
        if rule.endswith('.pt'):
            if device.type == 'cuda':
                model_CKPT = torch.load('./model/' + mod_files[i_rules])
            else:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location='cpu')
            print('\nloading checkpoint:', mod_files[i_rules])
            model.policy.load_state_dict(model_CKPT)  # 表示加载模型参数到当前模型中
            model.policy_old.load_state_dict(model_CKPT)  # 表示加载模型参数到当前模型中
        print('rule:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        makespans = []
        times = []
        energies = []
        penalties = []
        for i_ins in range(num_ins):
            test_file = data_path + test_files[i_ins]
            with open(test_file) as file_object:
                line = file_object.readlines()
                ins_num_jobs, ins_num_mas, _ = nums_detec(line)
            
            env_test_paras["num_jobs"] = ins_num_jobs
            env_test_paras["num_mas"] = ins_num_mas

            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
            # Create environment object
            else:
                # Clear the existing environment
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  # Get GPU memory info
                if meminfo.used / meminfo.total > 0.7:
                    envs.clear()
                # DRL-S, each env contains multiple (=num_sample) copies of one instance，表示每个环境包含一个实例的多个副本
                if test_paras["sample"]:
                    env = gym.make('fjsp-v0', case=[test_file] * test_paras["num_sample"], env_paras=env_test_paras, data_source='file')
                # DRL-G, each env contains one instance，表示每个环境包含一个实例
                else:
                    env = gym.make('fjsp-v0', case=[test_file], env_paras=env_test_paras, data_source='file')
                envs.append(copy.deepcopy(env))
                print("Create env[{0}]".format(i_ins))
            
            if rule == "FIFO":
                makespan, time_re, energy, penalty = schedule_fifo(env)
                makespans.append(torch.as_tensor(makespan, dtype=torch.float32))
                times.append(torch.as_tensor(time_re, dtype=torch.float32))
                energies.append(torch.as_tensor(energy, dtype=torch.float32))
                penalties.append(torch.as_tensor(penalty, dtype=torch.float32))
                continue
            elif rule == "MOR":
                makespan, time_re, energy, penalty = schedule_mor(env)
                makespans.append(torch.as_tensor(makespan, dtype=torch.float32))
                times.append(torch.as_tensor(time_re, dtype=torch.float32))
                energies.append(torch.as_tensor(energy, dtype=torch.float32))
                penalties.append(torch.as_tensor(penalty, dtype=torch.float32))
                continue
            elif rule == "MWKR":
                makespan, time_re, energy, penalty = schedule_mwkr(env)
                makespans.append(torch.as_tensor(makespan, dtype=torch.float32))
                times.append(torch.as_tensor(time_re, dtype=torch.float32))
                energies.append(torch.as_tensor(energy, dtype=torch.float32))
                penalties.append(torch.as_tensor(penalty, dtype=torch.float32))
                continue
            elif rule == "NSGA2":
                makespan, time_re, energy, penalty = schedule_nsga2(env, test_paras)
                makespans.append(torch.as_tensor(makespan, dtype=torch.float32))
                times.append(torch.as_tensor(time_re, dtype=torch.float32))
                energies.append(torch.as_tensor(energy, dtype=torch.float32))
                penalties.append(torch.as_tensor(penalty, dtype=torch.float32))
                continue

            # Schedule an instance/environment  
            # DRL-S
            if test_paras["sample"]:
                makespan, time_re, energy, penalty, best_idx = schedule(
                    env,
                    model,
                    memories,
                    flag_sample=test_paras["sample"],
                    reward_weights=reward_weights,
                )
                makespans.append(torch.as_tensor(makespan[best_idx], dtype=torch.float32))
                times.append(torch.as_tensor(time_re, dtype=torch.float32))
                energies.append(torch.as_tensor(energy[best_idx], dtype=torch.float32))
                penalties.append(torch.as_tensor(penalty[best_idx], dtype=torch.float32))
                # 将最优的调度方案输出到save/test_{0}/solution/文件夹中
                env.save_solution_to_file('{0}/solution/{1}_solution.txt'.format(save_path, test_files[i_ins][:-4]), best_idx)
            # DRL-G
            else:
                time_s = []
                makespan_s = []  # In fact, the results obtained by DRL-G do not change
                energy_s = []
                penalty_s = []
                for j in range(test_paras["num_average"]):
                    makespan, time_re, energy, penalty, best_idx = schedule(env, model, memories)
                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    energy_s.append(energy)
                    penalty_s.append(penalty)
                    env.reset()
                makespans.append(torch.mean(torch.tensor(makespan_s)))
                times.append(torch.mean(torch.tensor(time_s)))
                energies.append(torch.mean(torch.tensor(energy_s)))
                penalties.append(torch.mean(torch.tensor(penalty_s)))
            print("finish env {0}".format(i_ins))
        print("rule_spend_time: ", time.time() - step_time_last)

        # Save makespan and time data to files
        data = pd.DataFrame(torch.tensor(makespans).t().tolist(), columns=[rule])
        data.to_excel(writer, sheet_name='makespan', index=False, startcol=i_rules + 1)
        data = pd.DataFrame(torch.tensor(energies).t().tolist(), columns=[rule])
        data.to_excel(writer, sheet_name='energy', index=False, startcol=i_rules + 1)
        data = pd.DataFrame(torch.tensor(penalties).t().tolist(), columns=[rule])
        data.to_excel(writer, sheet_name='penalty', index=False, startcol=i_rules + 1)
        data = pd.DataFrame(torch.tensor(times).t().tolist(), columns=[rule])
        data.to_excel(writer, sheet_name='time', index=False, startcol=i_rules + 1)
        writer.save()
        writer.close()

        for env in envs:
            env.reset()

    print("total_spend_time: ", time.time() - start)

def schedule(env, model, memories, flag_sample=False, reward_weights=None):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()
    i = 0
    while ~done:
        i += 1
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        state, rewards, dones = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    # print("spend_time: ", spend_time)

    # Verify the solution
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    
    makespan_batch = copy.deepcopy(env.makespan_batch)
    energy_batch = copy.deepcopy(env.energy_batch)
    penalty_batch = copy.deepcopy(env.penalty_batch)
    best_idx = 0

    if flag_sample:
        makespan_tensor = torch.as_tensor(makespan_batch, dtype=torch.float32)
        energy_tensor = torch.as_tensor(energy_batch, dtype=torch.float32)
        penalty_tensor = torch.as_tensor(penalty_batch, dtype=torch.float32)

        eps = 1e-9
        reward_weights = reward_weights or {}
        w_makespan = float(reward_weights.get("makespan", 1.0))
        w_energy = float(reward_weights.get("energy", 1.0))
        w_penalty = float(reward_weights.get("penalty", 1.0))

        makespan_norm = makespan_tensor / (torch.min(makespan_tensor) + eps)
        energy_norm = energy_tensor / (torch.min(energy_tensor) + eps)
        penalty_norm = penalty_tensor / (torch.min(penalty_tensor) + eps)
        score = (
            w_makespan * makespan_norm
            + w_energy * energy_norm
            + w_penalty * penalty_norm
        ) / max(w_makespan + w_energy + w_penalty, eps)
        best_idx = int(torch.argmin(score).item())

    return makespan_batch, spend_time, energy_batch, penalty_batch, best_idx


def schedule_fifo(env):
    """
    FIFO rule:
    1) choose the earliest-start schedulable operation among all eligible jobs;
    2) for that operation, prefer currently idle machines with shortest processing time;
    3) if no candidate machine is idle now, choose the machine that becomes available earliest.
    """
    if env.batch_size != 1:
        raise ValueError("FIFO rule in test.py currently supports batch_size == 1 only.")

    done = bool(env.done_batch[0].item())
    last_time = time.time()

    while not done:
        cur_time = float(env.time[0].item())

        best_job = None
        best_ope = None
        best_ma = None
        best_start = float("inf")
        best_end = float("inf")

        for job in range(env.num_jobs):
            if bool(env.mask_job_finish_batch[0, job].item()) or bool(env.mask_job_procing_batch[0, job].item()):
                continue

            ope = int(env.ope_step_batch[0, job].item())
            if ope > int(env.end_ope_biases_batch[0, job].item()):
                continue

            proc_time_row = env.proc_times_batch[0, ope, :]
            candidate_mas = torch.nonzero(proc_time_row > 0, as_tuple=False).flatten()
            if candidate_mas.numel() == 0:
                continue

            ma_idle_mask = ~env.mask_ma_procing_batch[0, candidate_mas]
            if torch.any(ma_idle_mask):
                idle_mas = candidate_mas[ma_idle_mask]
                idle_proc = proc_time_row[idle_mas]
                pick_idx = int(torch.argmin(idle_proc).item())
                ma = int(idle_mas[pick_idx].item())
            else:
                cand_avail = env.machines_batch[0, candidate_mas, 1]
                pick_idx = int(torch.argmin(cand_avail).item())
                ma = int(candidate_mas[pick_idx].item())

            op_start_lb = float(env.feat_opes_batch[0, 5, ope].item())
            ma_avail = float(env.machines_batch[0, ma, 1].item())
            start_time = max(cur_time, op_start_lb, ma_avail)
            end_time = start_time + float(proc_time_row[ma].item())

            if (start_time < best_start) or (start_time == best_start and end_time < best_end):
                best_start = start_time
                best_end = end_time
                best_job = job
                best_ope = ope
                best_ma = ma

        if best_job is None:
            flag_trans_2_next_time = env.if_no_eligible()
            while ~((~((flag_trans_2_next_time == 0) & (~env.done_batch))).all()):
                env.next_time(flag_trans_2_next_time)
                flag_trans_2_next_time = env.if_no_eligible()
            done = bool(env.done_batch[0].item())
            continue

        actions = torch.tensor([[best_ope], [best_ma], [best_job]], dtype=torch.long, device=env.proc_times_batch.device)
        _, _, dones = env.step(actions)
        done = bool(dones[0].item())

    spend_time = time.time() - last_time

    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")

    makespan = float(copy.deepcopy(env.makespan_batch)[0].item())
    energy = float(copy.deepcopy(env.energy_batch)[0].item())
    penalty = float(copy.deepcopy(env.penalty_batch)[0].item())

    return makespan, spend_time, energy, penalty


def schedule_mor(env):
    """
    MOR rule:
    1) choose the executable operation whose job has the most remaining operations;
    2) executable means the operation has at least one idle candidate machine;
    3) for that operation, choose the idle machine with shortest processing time;
    4) if none is executable at current time, advance via environment next-event logic and reselect.
    """
    if env.batch_size != 1:
        raise ValueError("MOR rule in test.py currently supports batch_size == 1 only.")

    done = bool(env.done_batch[0].item())
    last_time = time.time()

    while not done:
        cur_time = float(env.time[0].item())

        best_job = None
        best_ope = None
        best_ma = None
        best_remain = -1
        best_start = float("inf")
        best_end = float("inf")

        for job in range(env.num_jobs):
            if bool(env.mask_job_finish_batch[0, job].item()) or bool(env.mask_job_procing_batch[0, job].item()):
                continue

            ope = int(env.ope_step_batch[0, job].item())
            end_ope = int(env.end_ope_biases_batch[0, job].item())
            if ope > end_ope:
                continue

            proc_time_row = env.proc_times_batch[0, ope, :]
            candidate_mas = torch.nonzero(proc_time_row > 0, as_tuple=False).flatten()
            if candidate_mas.numel() == 0:
                continue

            ma_idle_mask = ~env.mask_ma_procing_batch[0, candidate_mas]
            if not torch.any(ma_idle_mask):
                continue

            idle_mas = candidate_mas[ma_idle_mask]
            idle_proc = proc_time_row[idle_mas]
            pick_idx = int(torch.argmin(idle_proc).item())
            ma = int(idle_mas[pick_idx].item())

            remain_ops = end_ope - ope + 1
            start_time = cur_time
            end_time = start_time + float(proc_time_row[ma].item())

            if (
                (remain_ops > best_remain)
                or (remain_ops == best_remain and start_time < best_start)
                or (remain_ops == best_remain and start_time == best_start and end_time < best_end)
            ):
                best_remain = remain_ops
                best_start = start_time
                best_end = end_time
                best_job = job
                best_ope = ope
                best_ma = ma

        if best_job is None:
            # No operation can be executed at current time: move to next event time.
            flag_trans_2_next_time = env.if_no_eligible()
            while ~((~((flag_trans_2_next_time == 0) & (~env.done_batch))).all()):
                env.next_time(flag_trans_2_next_time)
                flag_trans_2_next_time = env.if_no_eligible()
            done = bool(env.done_batch[0].item())
            continue

        actions = torch.tensor([[best_ope], [best_ma], [best_job]], dtype=torch.long, device=env.proc_times_batch.device)
        _, _, dones = env.step(actions)
        done = bool(dones[0].item())

    spend_time = time.time() - last_time

    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")

    makespan = float(copy.deepcopy(env.makespan_batch)[0].item())
    energy = float(copy.deepcopy(env.energy_batch)[0].item())
    penalty = float(copy.deepcopy(env.penalty_batch)[0].item())

    return makespan, spend_time, energy, penalty


def schedule_mwkr(env):
    """
    MWKR rule:
     1) choose the executable operation whose job has the largest remaining work;
     2) remaining work is estimated by summing the average processing time of
         unscheduled operations in that job;
    3) executable means the operation has at least one idle candidate machine;
     4) for selected operation, choose the idle machine with shortest processing time;
     5) if none is executable at current time, advance to next event time and reselect.
    """
    if env.batch_size != 1:
        raise ValueError("MWKR rule in test.py currently supports batch_size == 1 only.")

    done = bool(env.done_batch[0].item())
    last_time = time.time()

    while not done:
        cur_time = float(env.time[0].item())

        best_job = None
        best_ope = None
        best_ma = None
        best_work = -1.0
        best_start = float("inf")
        best_end = float("inf")

        for job in range(env.num_jobs):
            if bool(env.mask_job_finish_batch[0, job].item()) or bool(env.mask_job_procing_batch[0, job].item()):
                continue

            ope = int(env.ope_step_batch[0, job].item())
            end_ope = int(env.end_ope_biases_batch[0, job].item())
            if ope > end_ope:
                continue

            # Remaining work is the sum of average processing times of unscheduled ops.
            remain_work = 0.0
            for op_idx in range(ope, end_ope + 1):
                op_proc = env.proc_times_batch[0, op_idx, :]
                valid_proc = op_proc[op_proc > 0]
                if valid_proc.numel() > 0:
                    remain_work += float(torch.mean(valid_proc).item())

            proc_time_row = env.proc_times_batch[0, ope, :]
            candidate_mas = torch.nonzero(proc_time_row > 0, as_tuple=False).flatten()
            if candidate_mas.numel() == 0:
                continue

            ma_idle_mask = ~env.mask_ma_procing_batch[0, candidate_mas]
            if not torch.any(ma_idle_mask):
                continue

            idle_mas = candidate_mas[ma_idle_mask]
            idle_proc = proc_time_row[idle_mas]
            pick_idx = int(torch.argmin(idle_proc).item())
            ma = int(idle_mas[pick_idx].item())

            start_time = cur_time
            end_time = start_time + float(proc_time_row[ma].item())

            if (
                (remain_work > best_work)
                or (remain_work == best_work and start_time < best_start)
                or (remain_work == best_work and start_time == best_start and end_time < best_end)
            ):
                best_work = remain_work
                best_start = start_time
                best_end = end_time
                best_job = job
                best_ope = ope
                best_ma = ma

        if best_job is None:
            # No operation can be executed at current time: move to next event time.
            flag_trans_2_next_time = env.if_no_eligible()
            while ~((~((flag_trans_2_next_time == 0) & (~env.done_batch))).all()):
                env.next_time(flag_trans_2_next_time)
                flag_trans_2_next_time = env.if_no_eligible()
            done = bool(env.done_batch[0].item())
            continue

        actions = torch.tensor([[best_ope], [best_ma], [best_job]], dtype=torch.long, device=env.proc_times_batch.device)
        _, _, dones = env.step(actions)
        done = bool(dones[0].item())

    spend_time = time.time() - last_time

    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")

    makespan = float(copy.deepcopy(env.makespan_batch)[0].item())
    energy = float(copy.deepcopy(env.energy_batch)[0].item())
    penalty = float(copy.deepcopy(env.penalty_batch)[0].item())

    return makespan, spend_time, energy, penalty


def _advance_to_next_event(env):
    flag_trans_2_next_time = env.if_no_eligible()
    while ~((~((flag_trans_2_next_time == 0) & (~env.done_batch))).all()):
        env.next_time(flag_trans_2_next_time)
        flag_trans_2_next_time = env.if_no_eligible()


def _decode_priority_policy(env, chromosome):
    """
    Decode a chromosome into one schedule and return objective values.
    chromosome: [job_priorities..., machine_tradeoff]
    """
    num_jobs = env.num_jobs
    priorities = chromosome[:num_jobs]
    # 0 -> focus on processing energy, 1 -> focus on processing time
    beta = float(np.clip(chromosome[num_jobs], 0.0, 1.0))

    done = bool(env.done_batch[0].item())

    while not done:
        cur_time = float(env.time[0].item())

        best_job = None
        best_ope = None
        best_ma = None
        best_priority = -float("inf")
        best_end = float("inf")

        for job in range(env.num_jobs):
            if bool(env.mask_job_finish_batch[0, job].item()) or bool(env.mask_job_procing_batch[0, job].item()):
                continue

            ope = int(env.ope_step_batch[0, job].item())
            end_ope = int(env.end_ope_biases_batch[0, job].item())
            if ope > end_ope:
                continue

            proc_time_row = env.proc_times_batch[0, ope, :]
            proc_energy_row = env.proc_energy_batch[0, ope, :]
            candidate_mas = torch.nonzero(proc_time_row > 0, as_tuple=False).flatten()
            if candidate_mas.numel() == 0:
                continue

            ma_idle_mask = ~env.mask_ma_procing_batch[0, candidate_mas]
            if not torch.any(ma_idle_mask):
                continue

            idle_mas = candidate_mas[ma_idle_mask]
            idle_proc = proc_time_row[idle_mas].float()
            idle_energy = proc_energy_row[idle_mas].float()

            proc_norm = (idle_proc - torch.min(idle_proc)) / (torch.max(idle_proc) - torch.min(idle_proc) + 1e-9)
            energy_norm = (idle_energy - torch.min(idle_energy)) / (torch.max(idle_energy) - torch.min(idle_energy) + 1e-9)
            score = beta * proc_norm + (1.0 - beta) * energy_norm
            pick_idx = int(torch.argmin(score).item())
            ma = int(idle_mas[pick_idx].item())

            end_time = cur_time + float(proc_time_row[ma].item())
            job_priority = float(priorities[job])

            if (job_priority > best_priority) or (job_priority == best_priority and end_time < best_end):
                best_priority = job_priority
                best_end = end_time
                best_job = job
                best_ope = ope
                best_ma = ma

        if best_job is None:
            _advance_to_next_event(env)
            done = bool(env.done_batch[0].item())
            continue

        actions = torch.tensor([[best_ope], [best_ma], [best_job]], dtype=torch.long, device=env.proc_times_batch.device)
        _, _, dones = env.step(actions)
        done = bool(dones[0].item())

    gantt_ok = bool(env.validate_gantt()[0])
    makespan = float(env.makespan_batch[0].item())
    energy = float(env.energy_batch[0].item())
    penalty = float(env.penalty_batch[0].item())

    if not gantt_ok:
        # Penalize infeasible schedules to steer NSGA2 away from them.
        makespan += 1e6
        energy += 1e6
        penalty += 1e6

    return np.array([makespan, energy, penalty], dtype=np.float64)


def _fast_non_dominated_sort(objs):
    n = objs.shape[0]
    dominates = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=np.int32)
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            p_dom_q = np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q])
            q_dom_p = np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p])
            if p_dom_q:
                dominates[p].append(q)
            elif q_dom_p:
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts


def _crowding_distance(front, objs):
    if len(front) == 0:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    distance = {idx: 0.0 for idx in front}
    num_obj = objs.shape[1]

    for m in range(num_obj):
        sorted_front = sorted(front, key=lambda idx: objs[idx, m])
        f_min = objs[sorted_front[0], m]
        f_max = objs[sorted_front[-1], m]
        distance[sorted_front[0]] = float("inf")
        distance[sorted_front[-1]] = float("inf")
        if f_max - f_min < 1e-12:
            continue
        for k in range(1, len(sorted_front) - 1):
            prev_v = objs[sorted_front[k - 1], m]
            next_v = objs[sorted_front[k + 1], m]
            distance[sorted_front[k]] += (next_v - prev_v) / (f_max - f_min)

    return distance


def _binary_tournament(rank, crowding):
    a = random.randrange(len(rank))
    b = random.randrange(len(rank))
    if rank[a] < rank[b]:
        return a
    if rank[b] < rank[a]:
        return b
    if crowding[a] > crowding[b]:
        return a
    if crowding[b] > crowding[a]:
        return b
    return a if random.random() < 0.5 else b


def schedule_nsga2(env, test_paras=None):
    """
    NSGA-II for three-objective optimization:
    makespan, total energy, earliness/tardiness penalty.
    """
    if env.batch_size != 1:
        raise ValueError("NSGA2 rule in test.py currently supports batch_size == 1 only.")

    test_paras = test_paras or {}
    nsga2_paras = test_paras.get("nsga2_paras", {})
    pop_size = int(nsga2_paras.get("pop_size", 24))   # 16
    generations = int(nsga2_paras.get("generations", 20))    #10
    mutation_sigma = float(nsga2_paras.get("mutation_sigma", 0.1))   #0.1
    mutation_prob = float(nsga2_paras.get("mutation_prob", 0.2))   #0.2
    crossover_prob = float(nsga2_paras.get("crossover_prob", 0.9))   #0.9
    verbose = bool(nsga2_paras.get("verbose", True))   # True
    log_interval = int(nsga2_paras.get("log_interval", 1))   #1

    pop_size = max(pop_size, 4)
    generations = max(generations, 1)
    log_interval = max(log_interval, 1)

    num_jobs = env.num_jobs
    dim = num_jobs + 1

    start_time = time.time()
    template_env = copy.deepcopy(env)

    population = np.random.rand(pop_size, dim)

    def evaluate_population(pop):
        vals = np.zeros((pop.shape[0], 3), dtype=np.float64)
        for i in range(pop.shape[0]):
            env_i = copy.deepcopy(template_env)
            vals[i] = _decode_priority_policy(env_i, pop[i])
        return vals

    objectives = evaluate_population(population)

    prev_min = np.min(objectives, axis=0)
    if verbose:
        init_fronts = _fast_non_dominated_sort(objectives)
        init_front_size = len(init_fronts[0]) if init_fronts else 0
        init_mean = np.mean(objectives, axis=0)
        print(
            "[NSGA2] Gen 0/{0} | Front0={1} | Best(mk,ene,pen)=({2:.3f}, {3:.3f}, {4:.3f}) "
            "| Mean=({5:.3f}, {6:.3f}, {7:.3f})".format(
                generations,
                init_front_size,
                prev_min[0], prev_min[1], prev_min[2],
                init_mean[0], init_mean[1], init_mean[2],
            )
        )

    for gen in range(1, generations + 1):
        fronts = _fast_non_dominated_sort(objectives)
        rank = np.full(pop_size, 10**9, dtype=np.int32)
        crowding = np.zeros(pop_size, dtype=np.float64)

        for front_idx, front in enumerate(fronts):
            for idx in front:
                rank[idx] = front_idx
            c = _crowding_distance(front, objectives)
            for idx, dist in c.items():
                crowding[idx] = dist

        offspring = np.zeros_like(population)
        for i in range(0, pop_size, 2):
            p1 = population[_binary_tournament(rank, crowding)]
            p2 = population[_binary_tournament(rank, crowding)]

            c1 = p1.copy()
            c2 = p2.copy()

            if random.random() < crossover_prob:
                alpha = np.random.rand(dim)
                c1 = alpha * p1 + (1.0 - alpha) * p2
                c2 = alpha * p2 + (1.0 - alpha) * p1

            if random.random() < mutation_prob:
                c1 += np.random.normal(0.0, mutation_sigma, size=dim)
            if random.random() < mutation_prob:
                c2 += np.random.normal(0.0, mutation_sigma, size=dim)

            offspring[i] = np.clip(c1, 0.0, 1.0)
            if i + 1 < pop_size:
                offspring[i + 1] = np.clip(c2, 0.0, 1.0)

        off_obj = evaluate_population(offspring)

        combined_pop = np.vstack([population, offspring])
        combined_obj = np.vstack([objectives, off_obj])

        fronts = _fast_non_dominated_sort(combined_obj)
        new_indices = []

        for front in fronts:
            if len(new_indices) + len(front) <= pop_size:
                new_indices.extend(front)
            else:
                c = _crowding_distance(front, combined_obj)
                front_sorted = sorted(front, key=lambda idx: c[idx], reverse=True)
                remain = pop_size - len(new_indices)
                new_indices.extend(front_sorted[:remain])
                break

        new_indices = np.array(new_indices, dtype=np.int64)
        population = combined_pop[new_indices]
        objectives = combined_obj[new_indices]

        if verbose and (gen % log_interval == 0 or gen == generations):
            cur_min = np.min(objectives, axis=0)
            cur_mean = np.mean(objectives, axis=0)
            cur_fronts = _fast_non_dominated_sort(objectives)
            cur_front_size = len(cur_fronts[0]) if cur_fronts else 0
            improve = prev_min - cur_min
            print(
                "[NSGA2] Gen {0}/{1} | Front0={2} | Best(mk,ene,pen)=({3:.3f}, {4:.3f}, {5:.3f}) "
                "| Mean=({6:.3f}, {7:.3f}, {8:.3f}) | dBest=({9:.3f}, {10:.3f}, {11:.3f})".format(
                    gen,
                    generations,
                    cur_front_size,
                    cur_min[0], cur_min[1], cur_min[2],
                    cur_mean[0], cur_mean[1], cur_mean[2],
                    improve[0], improve[1], improve[2],
                )
            )
            prev_min = cur_min

    final_fronts = _fast_non_dominated_sort(objectives)
    first_front = final_fronts[0] if final_fronts else list(range(pop_size))

    front_obj = objectives[first_front]
    obj_min = np.min(front_obj, axis=0)
    obj_max = np.max(front_obj, axis=0)
    norm_front = (front_obj - obj_min) / (obj_max - obj_min + 1e-9)
    compromise_idx = int(np.argmin(np.sum(norm_front, axis=1)))
    best_idx = first_front[compromise_idx]
    best_obj = objectives[best_idx]

    spend_time = time.time() - start_time
    return float(best_obj[0]), float(spend_time), float(best_obj[1]), float(best_obj[2])


if __name__ == '__main__':
    main()