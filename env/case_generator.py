import random


class CaseGenerator:
    '''
    FJSP instance generator
    '''
    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None, path='data_test1/1508/',
                 flag_same_opes=True, flag_doc=False):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # 表示是否保存生成的实例到文件中
        self.flag_same_opes = flag_same_opes  # 表示是否每个job的工序数量相同，True表示相同，False表示不同
        self.nums_ope = nums_ope  # 表示每个job的工序数量，长度为num_jobs
        self.path = path  # 保存生成的实例的路径
        self.job_init = job_init  # 表示初始的job数量，生成的实例中job数量为self.num_jobs，且当flag_same_opes为True时，每个job的工序数量相同
        self.num_mas = num_mas  # 表示机器数量

        self.mas_per_ope_min = 1  # 表示每个工序可选的最少机器数量
        self.mas_per_ope_max = num_mas
        self.opes_per_job_min = opes_per_job_min  # 表示每个job的最少工序数量
        self.opes_per_job_max = opes_per_job_max  # 表示每个job的最多工序数量
        self.proctime_per_ope_min = 1  # 表示每个工序的最小平均加工时间
        self.proctime_per_ope_max = 20  # 表示每个工序的最大平均加工时间
        self.proctime_dev = 0.2  # 表示加工时间的偏差
        self.seed = 42  # 随机种子

    def get_case(self, idx=0):
        '''
        生成FJSP实例
        :param idx: 实例编号
        '''

        random.seed(self.seed)
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]
        self.num_opes = sum(self.nums_ope)  # 表示所有job的工序总数量
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)] 
        self.num_options = sum(self.nums_option)  # 表示所有工序的可选机器总数量
        self.ope_ma = []   # The machine id of each operation
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas+1), val))  # 表示每个operation的可选machine id，且每个operation的可选machine id不重复
        self.proc_time = []
        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in range(self.num_opes)]  # 表示每个operation的平均加工时间
        for i in range(len(self.nums_option)):
            low_bound = max(self.proctime_per_ope_min,round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max,round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.proc_time = self.proc_time + proc_time_ope
        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes))
        adding_lines = []
        adding_lines_doc = []
        adding_lines1 = []
        adding_lines1_doc = []
        for i in range(self.num_jobs):

            earliness_cost = random.randint(1, 10)
            tardiness_cost = random.randint(1, 10)
            ready_time = random.randint(0, 20)
            weight = random.randint(1, 5)


            flag = 0   # flag to control the position in line
            flag_time = 0   # flag to control the position of processing time and machine id in line
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            adding_line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i]+self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    adding_line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i]+idx_ope])
                    adding_line.append(self.nums_option[self.num_ope_biass[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])  # 表示机器id
                    adding_line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])  # 表示加工时间
                    adding_line.append(random.randint(4, 15))   # 表示机器加工能耗
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)

                    adding_str_line = " ".join([str(val) for val in adding_line])
                    adding_lines.append(adding_str_line + '\n')
                    adding_lines_doc.append(adding_str_line)

                    # 计算该job的每个工序的平均加工时间之和
                    avg_proc_time = 0
                    for j in range(self.nums_ope[i]):
                        idx_ope_j = self.num_ope_biass[i] + j
                        avg_proc_time += self.proc_times_mean[idx_ope_j]
                    due_time = int(ready_time +  avg_proc_time * 1.5)
                    adding_lines1.append('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(weight, ready_time, due_time, earliness_cost, tardiness_cost))
                    adding_lines1_doc.append('{0}\t{1}\t{2}\t{3}\t{4}'.format(weight, ready_time, due_time, earliness_cost, tardiness_cost))
                    break
        lines.append('\n')
        adding_lines.append('\n')
        if self.flag_doc:
            doc = open(self.path + 'raw/{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx+1),3)),'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()

        machine_count = self.num_mas
        transport_time = [[0 for _ in range(machine_count)] for _ in range(machine_count)]  # 表示机器间运输时间矩阵
        transport_energy = [[0 for _ in range(machine_count)] for _ in range(machine_count)]  # 表示机器间运输能量矩阵
        for i in range(machine_count):
            for j in range(machine_count):
                if i < j:
                    temp_time = random.randint(1, 3)
                    temp_energy = random.randint(1,4)
                    transport_time[i][j] = temp_time
                    transport_time[j][i] = temp_time
                    transport_energy[i][j] = temp_energy
                    transport_energy[j][i] = temp_energy

        adding_lines.extend(adding_lines1)
        adding_lines_doc.extend(adding_lines1_doc)

        for i in range(machine_count):
            row = ' '.join(str(val) for val in transport_time[i])
            adding_lines.append(row + '\n')  # 表示机器间运输时间矩阵
            adding_lines_doc.append(row)  # 表示机器间运输时间矩阵
        for i in range(machine_count):
            row = ' '.join(str(val) for val in transport_energy[i])
            adding_lines.append(row + '\n')  # 表示机器间运输能量矩阵
            adding_lines_doc.append(row)  # 表示机器间运输能量矩阵
        
        if self.flag_doc:
            doc = open(self.path + 'add/{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx+1),3)),'a')
            for i in range(len(adding_lines_doc)):
                print(adding_lines_doc[i], file=doc)
            doc.close()
        
        self.seed += 1
        return lines, self.num_jobs, self.num_jobs, adding_lines
