import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def to_vector(k_neighbor_sa, ddl, total_access_num):
        max_len = 1000

        tmp_list = list(k_neighbor_sa)
        result_vector = np.zeros(max_len)
        ptr = 0
        for state, action in tmp_list:
            state_list = list(state)
            for e in state_list:
                result_vector[ptr] = e
                ptr += 1
            loc, dest = action
            result_vector[ptr + loc + 1] = 1
            ptr += ddl + 1
            result_vector[ptr + dest] = 1
            ptr += total_access_num
        return result_vector[:ptr]

#  Q 值的近似估计
class QFunction:
    def __init__(self, ddl, total_access_num, learning_rate, gamma, T):
        self.ddl = ddl
        self.total_access_num = total_access_num
        self.global_q = None
        self.optimizer = None
        self.learning_rate = learning_rate
        self.sa_list = []
        self.r_list = []
        self.gamma = gamma
        self.buffer_size = T 

    def get(self, k_neighbor_sa):
        if k_neighbor_sa is None:
            return 0.0
        elif self.global_q is None:
            inputDim = to_vector(k_neighbor_sa, self.ddl, self.total_access_num).shape[0]
            self.global_q = NeuralNetwork(inputDim)
            self.optimizer = optim.Adam(self.global_q.parameters(), lr=self.learning_rate)
            return 0.0
        else:
            s = torch.tensor(to_vector(k_neighbor_sa, self.ddl, self.total_access_num), dtype=torch.float)
            with torch.no_grad():
                result = self.global_q(s)
                return result.item()

    def get_list(self, sa_query_list):
        if self.global_q is None:
            lastStateAction = sa_query_list[0]
            inputDim = to_vector(lastStateAction, self.ddl, self.total_access_num).shape[0]
            self.global_q = NeuralNetwork(inputDim)
            self.optimizer = optim.Adam(self.global_q.parameters(), lr=self.learning_rate)

        input_list = []
        for sa in sa_query_list:
            tmp = to_vector(sa, self.ddl, self.total_access_num)
            input_list.append(tmp)

        sa_batch = torch.tensor(input_list, dtype=torch.float)
        with torch.no_grad():
            result_list = self.global_q(sa_batch)
            return result_list.numpy()

    def update_td1(self, last_state_action, current_state_action, last_reward):
        if self.global_q is None:
            inputDim = to_vector(last_state_action, self.ddl, self.total_access_num).shape[0]
            self.global_q = NeuralNetwork(inputDim)
            self.optimizer = optim.Adam(self.global_q.parameters(), lr=self.learning_rate)

        current_sa = to_vector(current_state_action, self.ddl, self.total_access_num)
        last_sa = to_vector(last_state_action, self.ddl, self.total_access_num)
        self.sa_list.append(last_sa)
        self.r_list.append(last_reward)
        if len(self.sa_list) < self.buffer_size:
            return

        final_sa = torch.tensor(current_sa, dtype=torch.float)
        R = self.global_q(final_sa).item()
        td_target_list = []
        for reward in self.r_list[::-1]:
            R = self.gamma * R + reward
            td_target_list.append([R])
        td_target_list.reverse()

        sa_batch = torch.tensor(self.sa_list, dtype=torch.float)
        td_target = torch.tensor(td_target_list, dtype=torch.float)

        loss = F.smooth_l1_loss(self.global_q(sa_batch), td_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.sa_list = []
        self.r_list = []

    def update_td0(self, last_state_action, current_state_action, last_reward):
        if self.global_q is None:
            inputDim = to_vector(last_state_action, self.ddl, self.total_access_num).shape[0]
            self.global_q = NeuralNetwork(inputDim)
            self.optimizer = optim.Adam(self.global_q.parameters(), lr=self.learning_rate)

        current_sa = to_vector(current_state_action, self.ddl, self.total_access_num)
        last_sa = to_vector(last_state_action, self.ddl, self.total_access_num)
        self.sa_list.append(last_sa)
        self.r_list.append(last_reward)
        if len(self.sa_list) < self.buffer_size:
            return

        evaluate_list = copy.deepcopy(self.sa_list[1:])
        evaluate_list.append(current_sa)
        eva_batch = torch.tensor(evaluate_list, dtype=torch.float)
        with torch.no_grad():
            td_target_list = self.global_q(eva_batch).numpy()
        for t in range(self.buffer_size):
            td_target_list[t, 0] = self.gamma * td_target_list[t, 0] + self.r_list[t]

        sa_batch = torch.tensor(self.sa_list, dtype=torch.float)
        td_target = torch.tensor(td_target_list, dtype=torch.float)

        loss = F.smooth_l1_loss(self.global_q(sa_batch), td_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.sa_list = []
        self.r_list = []

    