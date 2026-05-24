import torch
import numpy as np

def load_fjs(lines, num_mas, num_opes, adding_lines):
    '''
    Load the local FJSP instance.
    '''
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))  # 加工时间矩阵，行表示工序，列表示机器，元素值表示该工序在该机器上的加工时间，0表示该工序不能在该机器上加工
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)  # 前置工序矩阵，行表示工序，列表示工序，元素值表示该行工序是否是该列工序的前置工序
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()  # 累积加工时间矩阵，行表示工序，列表示工序，元素值表示该行工序的所有前置工序（包括直接前置和间接前置）的加工时间之和
    nums_ope = []  # A list of the number of operations for each job
    opes_appertain = np.array([])  # A list of the job id to which each operation belongs, and the job id starts from 0
    num_ope_biases = []  # The id of the first operation of each job
    # Parse data line by line
    for line in lines:
        # first line
        if flag == 0:
            flag += 1
        # last line
        elif line == "\n":
            break
        # other
        else:
            num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
            num_ope_biases.append(num_ope_bias)
            # Detect information of this job and return the number of operations
            num_ope = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
            nums_ope.append(num_ope)
            # nums_option = np.concatenate((nums_option, num_option))
            opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope)*(flag-1)))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    # Fill zero if the operations are insufficient (for parallel computation)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes-opes_appertain.size)))

    # 解析附加信息：加工能耗、工序/作业特征、运输时间与能耗矩阵
    matrix_proc_energy, job_features, transport_time, transport_energy = parse_additional(
        adding_lines, num_mas, num_opes, num_ope_biases, nums_ope
    )

    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), matrix_cal_cumul, matrix_proc_energy, \
           job_features, transport_time, transport_energy

def nums_detec(lines):  # 计算作业数、机器数和工序数
    '''
    Count the number of jobs, machines and operations
    '''
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i]!="\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes

def edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    '''
    Detect information of a job
    '''
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = np.array([])  # Store the number of processable machines for each operation of this job
    mac = 0
    for i in line_split:
        x = int(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            if idx_ope != num_ope-1:
                matrix_pre_proc[idx_ope+num_ope_bias][idx_ope+num_ope_bias+1] = True
            if idx_ope != 0:
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_ope+num_ope_bias-1] = 1
                matrix_cal_cumul[:, idx_ope+num_ope_bias] = matrix_cal_cumul[:, idx_ope+num_ope_bias-1]+vector
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope+num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    return num_ope

def energy_detec(line, num_ope_bias, matrix_proc_energy):
    '''
    Detect processing energy of a job
    '''
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0
    mac = 0
    for i in line_split:
        x = int(i)
        if flag == 0:
            num_ope = x
            flag += 1
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            flag += 1
        elif flag_time == 0:
            mac = x - 1
            flag += 1
            flag_time = 1
        else:
            matrix_proc_energy[idx_ope + num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    return num_ope

def parse_additional(adding_lines, num_mas, num_opes, num_ope_biases, nums_ope):
    '''
    Parse additional information: processing energy, job features, transport time and energy.
    '''
    matrix_proc_energy = torch.zeros(size=(num_opes, num_mas))
    job_features = None
    transport_time = None
    transport_energy = None

    if not adding_lines:
        return matrix_proc_energy, job_features, transport_time, transport_energy

    num_jobs = len(nums_ope)
    job_features = torch.zeros(size=(num_jobs, 5))

    def next_non_empty(lines, idx):
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        return idx

    idx = next_non_empty(adding_lines, 0)

    # 每个工序-机器可选项对应的加工能耗
    for j in range(num_jobs):
        idx = next_non_empty(adding_lines, idx)
        if idx >= len(adding_lines):
            break
        energy_detec(adding_lines[idx], num_ope_biases[j], matrix_proc_energy)
        idx += 1

    # 作业特征：权重、就绪时间、交期、提前惩罚、延期惩罚
    idx = next_non_empty(adding_lines, idx)
    for j in range(num_jobs):
        idx = next_non_empty(adding_lines, idx)
        if idx >= len(adding_lines):
            break
        parts = adding_lines[idx].split()
        if len(parts) >= 5:
            job_features[j] = torch.tensor([float(val) for val in parts[:5]])
        idx += 1

    # 运输时间矩阵与运输能耗矩阵
    idx = next_non_empty(adding_lines, idx)
    tokens = []
    for i in range(idx, len(adding_lines)):
        tokens.extend(adding_lines[i].split())
    values = [int(val) for val in tokens]
    mat_size = num_mas * num_mas
    if len(values) >= mat_size:
        time_vals = values[:mat_size]
        energy_vals = values[mat_size:mat_size * 2]
        if len(energy_vals) < mat_size:
            energy_vals = energy_vals + [0] * (mat_size - len(energy_vals))
        transport_time = torch.tensor(time_vals).reshape(num_mas, num_mas)
        transport_energy = torch.tensor(energy_vals).reshape(num_mas, num_mas)

    return matrix_proc_energy, job_features, transport_time, transport_energy