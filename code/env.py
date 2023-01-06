import numpy as np


class GlobalNetwork:
    def __init__(self, node_num, k):
        self.user_node_num = node_num
        # k-hop neighbours
        self.k = k
        self.adjacency_matrix = np.eye(self.user_node_num, dtype=int)
        self.adjacency_matrix_power = [np.eye(self.user_node_num, dtype=int)]
        self.neighbours = {}
        self.edges_added = False

    def add_edge(self, i, j):
        self.adjacency_matrix[i, j] = 1
        self.adjacency_matrix[j, i] = 1

    def finish_adding_edges(self):
        # for each hop, self.adjacency_matrix_power[d] stores the hop adjacency matrix
        for hop in range(self.k):
            tmp = np.matmul(np.eye(self.user_node_num, dtype=int), self.adjacency_matrix)
            self.adjacency_matrix_power.append(tmp)
        self.edges_added = True

    def find_neighbours(self, i, d):
        if not self.edges_added:
            return -1
        if (i, d) in self.neighbours:
            return self.neighbours[(i, d)]
        # 获取 d-hop neighbours 的 user i
        neighbours = []
        for j in range(self.user_node_num):
            if self.adjacency_matrix_power[d][i, j] > 0:
                neighbours.append(j)
        self.neighbours[(i, d)] = neighbours
        return neighbours

class AccessNetwork(GlobalNetwork):
    def __init__(self, node_num, k, access_num):
        super(AccessNetwork, self).__init__(node_num, k)
        self.access_num = access_num
        self.access_matrix = np.zeros((node_num, access_num), dtype=int)
        self.transmit_probability = np.ones(access_num)
        self.service_num = np.zeros(access_num, dtype=int)

    def add_access(self, i, a):
        self.access_matrix[i, a] = 1
        self.service_num[a] += 1

    def finish_adding_access(self):
        self.adjacency_matrix = np.matmul(self.access_matrix, np.transpose(self.access_matrix))
        super(AccessNetwork, self).finish_adding_edges()

    def find_access(self, i):
        access_points = []
        for j in range(self.access_num):
            if self.access_matrix[i, j] > 0:
                access_points.append(j)
        return access_points

class Env:
    def __init__(self, height, width, k, node_per_grid=1, transmit_prob='random', ddl=2, arrival_prob=0.5):
        self.height = height
        self.width = width
        self.k = k
        self.node_per_grid = node_per_grid
        self.transmit_probability = transmit_prob
        self.ddl = ddl
        self.arrival_prob = arrival_prob
        self.user_node_num = height * width * node_per_grid
        self.access_num = (height - 1) * (width - 1)
        self.global_state = np.zeros((self.user_node_num, self.ddl), dtype=int)
        self.new_global_state = np.zeros((self.user_node_num, self.ddl), dtype=int)
        self.global_action = np.zeros((self.user_node_num, 2), dtype=int)
        self.global_reward = np.zeros(self.user_node_num, dtype=float)
        self.accessNetwork = self.construct_grid_network(self.user_node_num, self.width, self.height, self.k, self.node_per_grid)

    def initialize(self):
        last = np.random.binomial(n=1, p=self.arrival_prob, size=self.user_node_num)
        self.global_state = np.zeros((self.user_node_num, self.ddl), dtype=int)
        self.global_state[:, self.ddl - 1] = last
        self.global_reward = np.zeros(self.user_node_num, dtype=float)

    def observe_state_g(self, index, depth):
        result = []
        for j in self.accessNetwork.find_neighbours(index, depth):
            result.append(tuple(self.global_state[j, :]))
        return tuple(result)

    def observe_state_action_g(self, index, depth):
        result = []
        for j in self.accessNetwork.find_neighbours(index, depth):
            result.append((tuple(self.global_state[j, :]), (self.global_action[j, 0], self.global_action[j, 1])))
        return tuple(result)

    def observe_reward(self, index):
        return self.global_reward[index]

    def generate_reward(self):
        self.global_reward = np.zeros(self.user_node_num, dtype=float)
        self.new_global_state = self.global_state
        client_counter = - np.ones(self.access_num, dtype=int)
        for i in range(self.user_node_num):
            access_point = self.global_action[i, 1]
            if access_point == -1:
                continue
            if client_counter[access_point] == -1:
                client_counter[access_point] = i
            elif client_counter[access_point] >= 0:
                client_counter[access_point] = -2
        for a in range(self.access_num):
            if client_counter[a] >= 0:
                client = client_counter[a]
                slot = self.global_action[client, 0]
                if self.global_state[client, slot] == 1:
                    success = np.random.binomial(1, self.accessNetwork.transmit_probability[a])
                    if success == 1:
                        self.new_global_state[client, slot] = 0
                        self.global_reward[client] = 1.0
        last = np.random.binomial(n=1, p=self.arrival_prob, size=self.user_node_num)
        self.new_global_state[:, 0:(self.ddl - 1)] = self.new_global_state[:, 1:self.ddl]
        self.new_global_state[:, self.ddl - 1] = last

    def construct_grid_network(self, node_num, width, height, k, node_per_grid):
        access_num = (width - 1) * (height - 1)
        access_network = AccessNetwork(node_num=node_num, k=k, access_num=access_num)
        for j in range(access_num):
            upper_left = j // (width - 1) * width + j % (width - 1)
            upper_right = upper_left + 1
            lower_left = upper_left + width
            lower_right = lower_left + 1
            for a in [upper_left, upper_right, lower_left, lower_right]:
                for b in range(node_per_grid):
                    access_network.add_access(node_per_grid * a + b, j)
        access_network.finish_adding_access()
        np.random.seed(0)
        transmit_prob = np.random.rand(access_num)
        access_network.transmit_probability = transmit_prob
        return access_network

    def step(self):
        self.global_state = self.new_global_state

    def update_action(self, index, action):
        slot, access_point = action
        self.global_action[index, 0] = slot
        self.global_action[index, 1] = access_point