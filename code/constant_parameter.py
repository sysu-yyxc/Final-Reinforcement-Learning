# 外层循环参数
out_loop = 12 # 120000

# 内层循环参数
in_loop = 5 # 16

# Scalable Actor Critic algorithm 的参数 kappa
k = 1

# 网络高度
height = 3

# 网络宽度
width = 4

# 论文中 nodes per grid 数量
node_per_grid = 2

# user nodes 数量
UserNodeNum = height * width * node_per_grid

# 折扣因子
gamma = 0.7

# 初始化的生命周期
d = 2

arrival_probability = 0.5

# 更新 policy 频率
EVALUATE_FREQ = 2000

RESET_FREQ = 100

