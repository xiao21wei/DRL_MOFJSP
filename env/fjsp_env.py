import sys
import os
import gym
import torch

from dataclasses import dataclass
from env.load_data import load_fjs, nums_detec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import copy
from utils.my_utils import read_json, write_json


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None   # shape: (batch_size, num_opes), the mapping between operations and jobs
    ope_pre_adj_batch: torch.Tensor = None      # shape: (batch_size, num_opes, num_opes), precedence adjacency matrix of operations
    ope_sub_adj_batch: torch.Tensor = None      # shape: (batch_size, num_opes, num_opes), subsequent adjacency matrix of operations
    end_ope_biases_batch: torch.Tensor = None   # shape: (batch_size, num_jobs), the id of the last operation of each job
    nums_opes_batch: torch.Tensor = None        # shape: (batch_size), the number of operations for each instance

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch:  torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time

def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    Convert job features into operation features (such as dimension)
    '''
    return feat_job_batch.gather(1, opes_appertain_batch)

class FJSPEnv(gym.Env):
    '''
    FJSP environment
    '''
    def __init__(self, case, env_paras, data_source='case'):
        '''
        :param case: The instance generator or the addresses of the instances
        :param env_paras: A dictionary of parameters for the environment
        :param data_source: Indicates that the instances came from a generator or files
        '''

        # load paras
        # static
        self.show_mode = env_paras["show_mode"]  # Result display mode (deprecated in the final experiment)
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs
        self.num_mas = env_paras["num_mas"]  # Number of machines
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch
        # load instance
        num_data = 12  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []
        lines_add = []
        if data_source=='case':  # Generate instances through generators
            for i in range(self.batch_size):
                line1, _, _, line2 = case.get_case(i)  # Generate an instance and save it
                lines.append(line1)  # Generate an instance and save it
                lines_add.append(line2)
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                # Records the maximum number of operations in the parallel instances
                self.num_opes = max(self.num_opes, num_opes)
        else:  # Load instances from files
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    lines.append(line)
                path = case[i].replace('raw', 'add')
                with open(path) as file_object:
                    line2 = file_object.readlines()
                    lines_add.append(line2)
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
        # load feats
        for i in range(self.batch_size):
            load_data = load_fjs(lines[i], num_mas, self.num_opes, lines_add[i])
            for j in range(num_data):
                tensors[j].append(load_data[j])

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas), 表示每个操作在每台机器上的加工时间，0表示不能加工
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas), 表示每个操作和每台机器的邻接关系，1表示邻接，0表示不邻接
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job, 计算每个作业路径上的累计量，1表示相邻，0表示不相邻
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()  # 表示每个操作所属的作业，值为作业id，从0开始
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # additional feats from case_generator
        # shape: (batch_size, num_opes, num_mas), 表示每个操作在每台机器上的加工能耗
        self.proc_energy_batch = torch.stack(tensors[8], dim=0).float()
        # shape: (batch_size, num_jobs, 5), 表示作业特征(权重、就绪时间、交期、提前惩罚、延期惩罚)
        self.job_feat_batch = torch.stack(tensors[9], dim=0).float() if tensors[9][0] is not None else None
        # shape: (batch_size, num_mas, num_mas), 表示运输时间矩阵
        self.transport_time_batch = torch.stack(tensors[10], dim=0).float() if tensors[10][0] is not None else None
        # shape: (batch_size, num_mas, num_mas), 表示运输能耗矩阵
        self.transport_energy_batch = torch.stack(tensors[11], dim=0).float() if tensors[11][0] is not None else None

        # Reward weights for multi-objective optimization (makespan, energy, penalty)
        reward_weights = env_paras.get("reward_weights", {})
        self.w_makespan = float(reward_weights.get("makespan", 1.0))
        self.w_energy = float(reward_weights.get("energy", 1.0))
        self.w_penalty = float(reward_weights.get("penalty", 1.0))


        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        '''
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                Start time
            ma:
                Number of neighboring operations
                Available time
                Utilization
        '''
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))  # 表示每个操作的特征，维度由paras["ope_feat_dim"]决定
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], num_mas))  # 表示每台机器的特征，维度由paras["ma_feat_dim"]决定

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)  # 表示每个操作的可选机器数量
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)  # 表示每个操作的平均加工时间
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)  # 表示每个操作所属作业的总操作数
        feat_opes_batch[:, 5, :] = torch.zeros_like(feat_opes_batch[:, 2, :])  # 表示每个操作的开始时间，初始值为0，后续根据就绪时间和前置工序的完成时间进行更新
        # 初始化工序开始时间（向量化）：首工序=ready_time；其余工序=前序累计(平均加工+平均运输)
        mean_transport_batch = self.transport_time_batch.mean(dim=(1, 2))
        ready_time_batch = self.job_feat_batch[:, :, 1]

        op_idx = torch.arange(self.num_opes, device=feat_opes_batch.device).unsqueeze(0).expand(self.batch_size, -1)
        valid_op_mask = op_idx < self.nums_opes.unsqueeze(1)

        start_idx_per_ope = convert_feat_job_2_ope(self.num_ope_biases_batch, self.opes_appertain_batch)
        ready_time_per_ope = convert_feat_job_2_ope(ready_time_batch, self.opes_appertain_batch)

        delta_time = feat_opes_batch[:, 2, :] + mean_transport_batch.unsqueeze(1)
        prefix_time = torch.cumsum(delta_time, dim=1)

        prev_idx = torch.clamp(op_idx - 1, min=0)
        prefix_prev = prefix_time.gather(1, prev_idx)

        start_prev_idx = torch.clamp(start_idx_per_ope - 1, min=0)
        prefix_start_prev = torch.where(
            start_idx_per_ope > 0,
            prefix_time.gather(1, start_prev_idx),
            torch.zeros_like(prefix_time)
        )

        offset = torch.where(
            op_idx > start_idx_per_ope,
            prefix_prev - prefix_start_prev,
            torch.zeros_like(prefix_time)
        )
        feat_opes_batch[:, 5, :] = (ready_time_per_ope + offset) * valid_op_mask.float()
        end_time_batch = (feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)  # 表示每个作业的完成时间（即最后一个操作的完成时间）
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)  # 表示每个操作所属作业的完成时间

        # 将新增作业特征映射为工序特征
        feat_opes_batch[:, 6, :] = convert_feat_job_2_ope(self.job_feat_batch[:, :, 0], self.opes_appertain_batch)  # 权重
        feat_opes_batch[:, 7, :] = convert_feat_job_2_ope(self.job_feat_batch[:, :, 1], self.opes_appertain_batch)  # 就绪时间
        feat_opes_batch[:, 8, :] = convert_feat_job_2_ope(self.job_feat_batch[:, :, 2], self.opes_appertain_batch)  # 交期
        feat_opes_batch[:, 9, :] = convert_feat_job_2_ope(self.job_feat_batch[:, :, 3], self.opes_appertain_batch)  # 提前惩罚
        feat_opes_batch[:, 10, :] = convert_feat_job_2_ope(self.job_feat_batch[:, :, 4], self.opes_appertain_batch)  # 延期惩罚

        # 预估交付提前/延期成本：weight * (earliness * early_penalty + tardiness * tardy_penalty)
        earliness = torch.clamp(feat_opes_batch[:, 8, :] - feat_opes_batch[:, 4, :], min=0.0)
        tardiness = torch.clamp(feat_opes_batch[:, 4, :] - feat_opes_batch[:, 8, :], min=0.0)
        feat_opes_batch[:, 11, :] = feat_opes_batch[:, 6, :] * (feat_opes_batch[:, 9, :] * earliness + feat_opes_batch[:, 10, :] * tardiness)

        # 预估工序加工完成能耗（包含运输能耗和加工能耗）
        mean_proc_energy_batch = torch.sum(self.proc_energy_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        mean_transport_energy_batch = self.transport_energy_batch.mean(dim=(1, 2))
        # 向量化计算：累计加工能耗 + 工序偏移量 * 平均运输能耗
        proc_prefix = torch.cumsum(mean_proc_energy_batch, dim=1)
        start_prev_idx = torch.clamp(start_idx_per_ope - 1, min=0)
        proc_prefix_start_prev = torch.where(
            start_idx_per_ope > 0,
            proc_prefix.gather(1, start_prev_idx),
            torch.zeros_like(proc_prefix)
        )
        proc_within_job = proc_prefix - proc_prefix_start_prev
        ope_offset = torch.clamp(op_idx - start_idx_per_ope, min=0).float()   # 表示每个操作在其所属作业中的偏移量，首工序偏移量为0，依次递增
        feat_opes_batch[:, 12, :] = (
            proc_within_job + ope_offset * mean_transport_energy_batch.unsqueeze(1)
        ) * valid_op_mask.float()  # 表示每个操作的完成能耗，包含运输能耗和加工能耗
        # 预估每个工序所属工件的总完工能耗（取该工件最后一道工序的累计能耗）
        end_energy_batch = feat_opes_batch[:, 12, :].gather(1, self.end_ope_biases_batch)  # 表示每个作业的完成能耗（即最后一个操作的完成能耗）
        feat_opes_batch[:, 13, :] = convert_feat_job_2_ope(end_energy_batch, self.opes_appertain_batch)  # 表示每个操作所属作业的完成能耗
        
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)  # 表示每台机器的邻接操作数量
        # 将运输时间与能耗作为机器特征（按行均值汇总）
        feat_mas_batch[:, 3, :] = self.transport_time_batch.mean(dim=2)
        feat_mas_batch[:, 4, :] = self.transport_energy_batch.mean(dim=2)

        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        self.predicted_makespan = self.feat_opes_batch[:, 4, :]  # 预估的每个作业的完成时间（即最后一个操作的完成时间）
        self.predicted_penalty = self.feat_opes_batch[:, 11, :]  # 预估的每个作业的提前/延期惩罚
        self.predicted_energy = self.feat_opes_batch[:, 13, :]  # 预估的每个作业的完成能耗（即最后一个操作的完成能耗）

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process, 表示每个作业是否正在加工
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)  # 表示每个作业是否正在加工
        # shape: (batch_size, num_jobs), True for completed jobs, 表示每个作业是否完成
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)  # 表示每个作业是否完成
        # shape: (batch_size, num_mas), True for machines in process  # 表示每台机器是否正在加工
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False)  # 表示每台机器是否正在加工
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))  # 表示每个操作的部分调度信息，维度为4，分别表示状态、分配的机器、开始时间和结束时间
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]  # 表示每个操作的开始时间，初始值为每个操作的开始时间，后续根据调度情况进行更新
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]  # 表示每个操作的结束时间，初始值为每个操作的所有前置工序（包括直接前置和间接前置）的加工时间之和加上平均加工时间
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))  # 表示每台机器是否空闲，1表示空闲，0表示忙碌

        # Makespan from real schedule (includes ready/transport effects)
        self.makespan_batch = torch.max(self.schedules_batch[:, :, 3], dim=1)[0]  # 表示每个实例的当前最大完成时间（即当前的makespan）
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # 表示每个实例是否完成，只有当所有作业都完成时才算完成

        # Cumulative metrics for logging and reward computation
        self.energy_batch = torch.zeros(self.batch_size)
        self.penalty_batch = torch.zeros(self.batch_size)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)

    def step(self, actions):  # 表示环境的转移函数，根据输入的动作更新环境状态，并计算奖励和完成标志
        '''
        Environment transition function
        '''
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1

        # Removed unselected O-M arcs of the scheduled operations，移除被调度的操作与未被调度的机器之间的邻接关系
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'，更新工序特征中与O-M邻接关系相关的部分，如状态、可选机器数量和加工时间
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack((torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                                                                        torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                                                                        proc_times), dim=1)  # 表示每个被调度的操作的状态、可选机器数量和加工时间，其中状态设为1，表示已调度
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1, opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0  # 表示每个被调度的操作的前一个操作与后续操作之间的累计量设为0，因为前一个操作已经完成，不再对后续操作产生影响

        # Update 'Number of unscheduled operations in the job'
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i]+1] -= 1

        # Update 'Start time' and 'Job completion time' (vectorized)
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled
        un_scheduled = 1 - is_scheduled
        estimate_times = torch.bmm(
            (start_times + mean_proc_time).unsqueeze(1),
            self.cal_cumul_adj_batch[self.batch_idxes, :, :]
        ).squeeze(1) * un_scheduled
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        selected_start_times = self.feat_opes_batch[self.batch_idxes, 5, opes]

        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                          self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[self.batch_idxes, :])

        # Update estimated earliness/tardiness cost for affected instances
        feat_opes_sel = self.feat_opes_batch[self.batch_idxes, :, :]  # 表示被调度的操作的特征，形状为(batch_size, ope_feat_dim, num_opes)
        due_minus_end = feat_opes_sel[:, 8, :] - feat_opes_sel[:, 4, :]  # 表示每个操作所属作业的交期与完成时间之差，正值表示提前，负值表示延期
        earliness = torch.relu(due_minus_end)  # 表示每个操作所属作业的提前时间，只有当交期大于完成时间时才有意义
        tardiness = torch.relu(-due_minus_end)  # 表示每个操作所属作业的延期时间，只有当完成时间大于交期时才有意义
        self.feat_opes_batch[self.batch_idxes, 11, :] = feat_opes_sel[:, 6, :] * (
            feat_opes_sel[:, 9, :] * earliness +
            feat_opes_sel[:, 10, :] * tardiness
        )

        # Update cumulative energy for affected instances
        mean_transport_energy_batch = self.transport_energy_batch.mean(dim=(1, 2))

        proc_energy_masked = self.proc_energy_batch * self.ope_ma_adj_batch.float()
        adj_count = torch.count_nonzero(self.ope_ma_adj_batch, dim=2).float()
        mean_proc_energy_batch = torch.sum(proc_energy_masked, dim=2).div(adj_count + 1e-9)

        b_idxes = self.batch_idxes
        bsz = b_idxes.size(0)
        num_opes = self.num_opes
        device = self.feat_opes_batch.device

        op_idx = torch.arange(num_opes, device=device).unsqueeze(0).expand(bsz, -1)
        valid_op_mask = op_idx < self.nums_opes[b_idxes].unsqueeze(1)

        is_scheduled = self.feat_opes_batch[b_idxes, 0, :] > 0
        ma_choice = torch.argmax(self.ope_ma_adj_batch[b_idxes, :, :], dim=2)

        proc_energy_real = self.proc_energy_batch[b_idxes, :, :].gather(2, ma_choice.unsqueeze(-1)).squeeze(-1)
        proc_energy_est = mean_proc_energy_batch[b_idxes, :]
        proc_energy = torch.where(is_scheduled, proc_energy_real, proc_energy_est)

        start_idx_per_ope = convert_feat_job_2_ope(self.num_ope_biases_batch[b_idxes, :], self.opes_appertain_batch[b_idxes, :])
        non_first_mask = op_idx > start_idx_per_ope

        prev_scheduled = torch.roll(is_scheduled, shifts=1, dims=1)
        prev_scheduled[:, 0] = False

        prev_ma = torch.roll(ma_choice, shifts=1, dims=1)
        trans_energy = mean_transport_energy_batch[b_idxes].unsqueeze(1).expand(-1, num_opes)

        transport_sub = self.transport_energy_batch[b_idxes, :, :]
        batch_grid = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, num_opes)
        trans_energy_real = transport_sub[batch_grid, prev_ma, ma_choice]

        use_real_trans = non_first_mask & is_scheduled & prev_scheduled & valid_op_mask
        trans_energy = torch.where(use_real_trans, trans_energy_real, trans_energy)

        delta_energy = proc_energy + torch.where(non_first_mask, trans_energy, torch.zeros_like(trans_energy))
        delta_energy = delta_energy * valid_op_mask.float()

        prefix_energy = torch.cumsum(delta_energy, dim=1)
        start_prev_idx = torch.clamp(start_idx_per_ope - 1, min=0)
        prefix_start_prev = torch.where(
            start_idx_per_ope > 0,
            prefix_energy.gather(1, start_prev_idx),
            torch.zeros_like(prefix_energy)
        )
        self.feat_opes_batch[b_idxes, 12, :] = (prefix_energy - prefix_start_prev) * valid_op_mask.float()

        end_energy_batch = self.feat_opes_batch[self.batch_idxes, 12, :].gather(1, self.end_ope_biases_batch[self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 13, :] = convert_feat_job_2_ope(end_energy_batch, self.opes_appertain_batch[self.batch_idxes, :])

        # Update partial schedule (state)
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas), dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
        self.machines_batch[self.batch_idxes, mas, 1] = selected_start_times + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        # Update feature vectors of machines
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = selected_start_times + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch==self.end_ope_biases_batch+1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        # 当前状态下的三个目标值（越小越好）：最大完工时间、总能耗、总提前/延期惩罚
        makespan_max = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        energy_max = self.feat_opes_batch[:, 13, :].gather(1, self.end_ope_biases_batch).sum(dim=1)
        penalty_max = self.feat_opes_batch[:, 11, :].gather(1, self.end_ope_biases_batch).sum(dim=1)

        # 计算相对改进量：(上一步 - 当前步) / 上一步。
        # 若当前值变小（目标改善）则 delta > 0；若变大（目标恶化）则 delta < 0。
        delta_makespan = (self.makespan_batch - makespan_max) / (self.makespan_batch + 1e-9)
        delta_energy = (self.energy_batch - energy_max) / (self.energy_batch + 1e-9)
        delta_penalty = (self.penalty_batch - penalty_max) / (self.penalty_batch + 1e-9)

        # 多目标加权平均奖励：按 makespan/energy/penalty 的权重融合三个相对改进量
        self.reward_batch = (self.w_makespan * delta_makespan + self.w_energy * delta_energy + self.w_penalty * delta_penalty) / (self.w_makespan + self.w_energy + self.w_penalty)

        # 将当前目标值保存为“上一步”基线，供下一步计算相对改进量
        self.makespan_batch = makespan_max
        self.energy_batch = energy_max
        self.penalty_batch = penalty_max

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time==0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

        # Update the vector for uncompleted instances
        mask_finish = (self.N+1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
            self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch, self.mask_ma_procing_batch,
                          self.ope_step_batch, self.time)
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                        self.proc_times_batch.size(2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # need to transit
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # remain available_time greater than current time
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        # Return the minimum value of available_time (the time to transit to)
        c = torch.min(b, dim=1)[0]
        # Detect the machines that completed (at above time)
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        return self.state

    def render(self, mode='human'):  # 表示环境的渲染函数，根据当前的部分调度信息绘制甘特图，供可视化使用
        '''
        Deprecated in the final experiment
        '''
        if self.show_mode == 'draw':
            num_jobs = self.num_jobs
            num_mas = self.num_mas
            print(sys.argv[0])
            color = read_json("./utils/color_config")["gantt_color"]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                          range(num_append_color)]
            write_json({"gantt_color": color}, "./utils/color_config")
            for batch_id in range(self.batch_size):
                schedules = self.schedules_batch[batch_id].to('cpu')
                fig = plt.figure(figsize=(10, 6))
                fig.canvas.set_window_title('Visual_gantt')
                axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
                y_ticks = []
                y_ticks_loc = []
                for i in range(num_mas):
                    y_ticks.append('Machine {0}'.format(i))
                    y_ticks_loc.insert(0, i + 1)
                labels = [''] * num_jobs
                for j in range(num_jobs):
                    labels[j] = "job {0}".format(j + 1)
                patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
                axes.cla()
                axes.set_title(u'FJSP Schedule')
                axes.grid(linestyle='-.', color='gray', alpha=0.2)
                axes.set_xlabel('Time')
                axes.set_ylabel('Machine')
                axes.set_yticks(y_ticks_loc, y_ticks)
                axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
                axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
                for i in range(int(self.nums_opes[batch_id])):
                    id_ope = i
                    idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                    id_machine = schedules[id_ope][1]
                    axes.barh(id_machine,
                             0.2,
                             left=schedules[id_ope][2],
                             color='#b2b2b2',
                             height=0.5)
                    axes.barh(id_machine,
                             schedules[id_ope][3] - schedules[id_ope][2] - 0.2,
                             left=schedules[id_ope][2]+0.2,
                             color=color[idx_job],
                             height=0.5)
                plt.show()
        return

    def get_idx(self, id_ope, batch_id):
        '''
        Get job and operation (relative) index based on instance index and operation (absolute) index
        '''
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2]-ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch

    def close(self):
        pass

    def save_solution_to_file(self, file_path, best_idx):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        best_idx = int(best_idx)
        schedules = self.schedules_batch[best_idx].to('cpu')
        num_mas = self.num_mas
        num_opes = int(self.nums_opes[best_idx].item())
        machine_ops = [[] for _ in range(num_mas)]
        for ope_id in range(num_opes):
            machine_id = int(schedules[ope_id][1])
            start_time = schedules[ope_id][2].item()
            job_id, ope_idx = self.get_idx(ope_id, best_idx)
            machine_ops[machine_id].append((job_id, ope_idx, start_time))
        for ops in machine_ops:
            ops.sort(key=lambda x: x[2])
        with open(file_path, 'w') as f:
            f.write(f"{num_mas}\n")
            for ops in machine_ops:
                f.write(f"{len(ops)} ")
                line = ' '.join(f"{job_id} {ope_idx}" for job_id, ope_idx, _ in ops)
                f.write(f"{line}\n")
        