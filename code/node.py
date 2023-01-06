class UserNode:
    def __init__(self, index):
        # node i
        self.index = index
        self.state = []
        self.action = []
        self.reward = []
        self.current_time_step = 0
        self.parameters = {}
        self.k_hop = []

    # 获取在time_step时刻节点的状态
    def get_state(self, time_step):
        if time_step <= len(self.state) - 1:
            return self.state[time_step]
        else:
            return -1

    # 获取在time_step时刻节点的动作
    def get_action(self, time_step):
        if time_step <= len(self.action) - 1:
            return self.action[time_step]
        else:
            return -1

    # 获取在time_step时刻节点的奖励
    def get_reward(self, time_step):
        if time_step <= len(self.reward) - 1:
            return self.reward[time_step]
        else:
            return -1

    # 论文算法，获取节点 k 领域的其他节点
    def get_k_hop_state_action(self, time_step):
        if time_step <= len(self.k_hop) - 1:
            return self.k_hop[time_step]
        else:
            return -1

    def reset(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.k_hop.clear()
        self.current_time_step = 0