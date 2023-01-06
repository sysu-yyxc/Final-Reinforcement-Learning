from scipy import special
import numpy as np
from network import QFunction
from constant_parameter import *
from node import UserNode


class Agent(UserNode):
    def __init__(self, index, deadline, arrival_prob, gamma, k, in_loop, env):
        super(Agent, self).__init__(index)
        self.ddl = deadline
        self.k = k
        self.arrival_prob = arrival_prob
        self.gamma = gamma
        self.packet_queue = np.zeros(self.ddl, dtype=int)
        self.access_points = env.accessNetwork.find_access(i=index)
        self.access_num = len(self.access_points)
        self.action_num = self.access_num * self.ddl + 1
        self.action_list = [(-1, -1)]
        for slot in range(self.ddl):
            for point in self.access_points:
                self.action_list.append((slot, point))
        # self.approximator = QApproximator(self.ddl, (height - 1) * (width - 1), 2e-4, gamma, in_loop)
        self.env = env
        self.approximator = QFunction(self.ddl, (height - 1) * (width - 1), 2e-4, gamma, in_loop)

    def get_q(self, state_action):
        return self.approximator.get(state_action)

    def get_q_list(self, sa_query_list):
        # return self.approximator.get_list(sa_query_list)
        return self.approximator.get_list(sa_query_list)

    def initialize_state(self):
        self.state.append(self.env.observe_state_g(self.index, 0)[0])

    def update_state(self):
        self.current_time_step += 1
        self.state.append(self.env.observe_state_g(self.index, 0)[0])
        self.packet_queue = self.state[-1]

    def update_reward(self):
        current_reward = self.env.observe_reward(self.index)
        self.reward.append(current_reward)

    def update_k_hop(self):
        self.k_hop.append(self.env.observe_state_action_g(self.index, self.k))

    def update_q_value(self):
        last_state_action = self.k_hop[-1]
        current_state_action = self.env.observe_state_action_g(self.index, self.k)
        # 更新 Q 函数权重值
        self.approximator.update_td0(last_state_action, current_state_action, self.reward[-2])
        if len(self.k_hop) == 0:
            self.k_hop.append(last_state_action)
        self.k_hop.append(current_state_action)

    def update_action(self, benchmark_policy=None):
        # check benchmark_policy
        if benchmark_policy is not None:
            action_prob = benchmark_policy[0]
            action_flag = np.random.binomial(1, action_prob)
            if action_flag == 0:
                self.action.append((-1, -1))
                self.env.update_action(self.index, (-1, -1))
                return
            bench_slot = -1
            for i in range(self.ddl):
                if self.packet_queue[i] > 0:
                    bench_slot = i
                    break
            if bench_slot == -1:
                self.action.append((-1, -1))
                self.env.update_action(self.index, (-1, -1))
                return
            # 选择 access point
            bench_prob = benchmark_policy[1:]
            bench_access_point = self.access_points[np.random.choice(a=self.access_num, p=bench_prob)]
            self.action.append((bench_slot, bench_access_point))
            self.env.update_action(self.index, (bench_slot, bench_access_point))
            return
        # 获得当前状态
        current_state = self.state[-1]
        # get the params based on the current state
        params = self.parameters.get(current_state, np.zeros(self.action_num))
        # 计算概率
        prob_vector = special.softmax(params)
        # 基于概率随机选择
        current_action = self.action_list[np.random.choice(a=self.action_num, p=prob_vector)]
        self.action.append(current_action)
        self.env.update_action(self.index, current_action)
        
    def update_parameters(self, k_hop_neighbors, eta):
        mutiplier1 = np.zeros(self.current_time_step + 1)
        for neighbor in k_hop_neighbors:
            neighbor_sa_list = []
            for t in range(self.current_time_step + 1):
                neighbor_sa_list.append(neighbor.get_k_hop_state_action(t))
            neighborQ_list = neighbor.get_q_list(neighbor_sa_list)
            mutiplier1 += neighborQ_list.flatten()
        for t in range(self.current_time_step + 1):
            mutiplier1[t] *= pow(self.gamma, t)
            mutiplier1[t] /= UserNodeNum

        for t in range(self.current_time_step + 1):
            current_state = self.state[t]
            current_action = self.action[t]
            params = self.parameters.get(current_state, np.zeros(self.action_num))
            probVec = special.softmax(params)
            grad = -probVec
            aaction_index = self.action_list.index(current_action)
            grad[aaction_index] += 1.0
            self.parameters[current_state] = params + eta * mutiplier1[t] * grad

    def total_reward(self):
        result = 0.0
        for t in range(self.current_time_step):
            result += (pow(self.gamma, t) * self.reward[t])
        return result