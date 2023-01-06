from constant_parameter import *
import numpy as np

# 评估 User Nodes
def evaluate_policy(nodes, rounds, env):
    total_reward = 0.0
    for _ in range(rounds):
        env.initialize()
        for i in range(UserNodeNum):
            nodes[i].reset()
            nodes[i].initialize_state()
        for i in range(UserNodeNum):
            nodes[i].update_action()
        env.generate_reward()
        for i in range(UserNodeNum):
            nodes[i].update_reward()
        for __ in range(1, in_loop + 1):
            env.step()
            for i in range(UserNodeNum):
                nodes[i].update_state()
            for i in range(UserNodeNum):
                nodes[i].update_action()
            env.generate_reward()
            for i in range(UserNodeNum):
                nodes[i].update_reward()
        average_reward = 0.0
        for i in range(UserNodeNum):
            average_reward += nodes[i].total_reward()
        average_reward /= UserNodeNum
        total_reward += average_reward
    total_reward = total_reward / rounds
    return total_reward

# 评估 benchmark（ALOHA protocol）
def evaluate_benchmark(nodes, rounds, action_probability, env):
    total_reward = 0.0
    benchmark_policys = []
    for i in range(UserNodeNum):
        access_points = env.accessNetwork.find_access(i)
        benchmark_policy = np.zeros(len(access_points) + 1)
        total_sum = 0.0
        for j in range(len(access_points)):
            tmp = 100 * env.accessNetwork.transmit_probability[access_points[j]] / env.accessNetwork.service_num[access_points[j]]
            total_sum += 100 * env.accessNetwork.transmit_probability[access_points[j]] / env.accessNetwork.service_num[access_points[j]]
            benchmark_policy[j + 1] = tmp
        # 平均 reward
        for j in range(len(access_points)):
            benchmark_policy[j + 1] /= total_sum
        # 动作概率
        benchmark_policy[0] = action_probability
        benchmark_policys.append(benchmark_policy)


    for _ in range(rounds):
        env.initialize()
        for i in range(UserNodeNum):
            nodes[i].reset()
            nodes[i].initialize_state()
        for i in range(UserNodeNum):
            nodes[i].update_action(benchmark_policys[i])
        env.generate_reward()
        for i in range(UserNodeNum):
            nodes[i].update_reward()
        for __ in range(1, in_loop + 1):
            env.step()
            for i in range(UserNodeNum):
                nodes[i].update_state()
            for i in range(UserNodeNum):
                nodes[i].update_action(benchmark_policys[i])
            env.generate_reward()
            for i in range(UserNodeNum):
                nodes[i].update_reward()
        average_reward = 0.0
        for i in range(UserNodeNum):
            average_reward += nodes[i].total_reward()
        average_reward /= UserNodeNum
        total_reward += average_reward
    total_reward = total_reward / rounds
    return total_reward