import math
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from constant_parameter import *
from evaluate import *
from env import *
from agent import Agent


if __name__ == "__main__":
    # 初始化环境
    env = Env(height=height, width=width, k=k, node_per_grid=node_per_grid)
    user_nodes = []
    for i in range(UserNodeNum):
        user_nodes.append(Agent(index=i, deadline=d, arrival_prob=arrival_probability, gamma=gamma, k=k, in_loop=in_loop, env=env))
    policy_rewards = []
    best_benchmark = 0.0
    best_benchmark_probability = 0.0
    # Scalable Actor Critic algorithm
    for m in trange(out_loop):
        if m == 0:
            policy_rewards.append(evaluate_policy(nodes=user_nodes, rounds=400, env=env))
            for i in range(20):
                tmp = evaluate_benchmark(nodes=user_nodes, rounds=100, action_probability=i / 20.0, env=env)
                if tmp > best_benchmark:
                    best_benchmark = tmp
                    best_benchmark_probability = i / 20.0
        env.initialize()
        for i in range(UserNodeNum):
            user_nodes[i].reset()
            user_nodes[i].initialize_state()
        for i in range(UserNodeNum):
            user_nodes[i].update_action()
        env.generate_reward()
        for i in range(UserNodeNum):
            user_nodes[i].update_reward()
        for i in range(UserNodeNum):
            user_nodes[i].update_k_hop()
        for t in range(1, in_loop + 1):
            env.step()
            for i in range(UserNodeNum):
                user_nodes[i].update_state()
            for i in range(UserNodeNum):
                user_nodes[i].update_action()
            env.generate_reward()
            for i in range(UserNodeNum):
                user_nodes[i].update_reward()
            for i in range(UserNodeNum):
                user_nodes[i].update_q_value()

        for i in range(UserNodeNum):
            neighbours = []
            for j in env.accessNetwork.find_neighbours(i, k):
                neighbours.append(user_nodes[j])
            user_nodes[i].update_parameters(neighbours, 5.0 / math.sqrt(m % RESET_FREQ + 1))

        if m % EVALUATE_FREQ == EVALUATE_FREQ - 1:
            policy_rewards.append(evaluate_policy(nodes=user_nodes, rounds=400, env=env))


    x = np.linspace(0, (len(policy_rewards) - 1) * EVALUATE_FREQ, len(policy_rewards))
    plt.plot(x, policy_rewards, label="Scalable Actor Critic")
    plt.hlines(y=best_benchmark, xmin=0, xmax=out_loop, colors='g', label="Benchmark")
    plt.xlabel('Number of Outer Loops')
    plt.ylabel('Discounted Reward')
    plt.legend(loc=4)
    plt.savefig("./img/h{}-w{}-c{}.jpg".format(height, width, node_per_grid))