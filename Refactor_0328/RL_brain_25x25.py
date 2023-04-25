import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

import copy

np.random.seed(1)
torch.manual_seed(1)


# 定义网络结构
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(Net, self).__init__()
        # nn.Linear(in_feature, out_feature)  全连接层
        # n_feature 指observation的size, n_hidden = 20  n_output = n_action
        self.fc1 = nn.Linear(n_feature, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc4 = nn.Linear(n_hidden3, n_hidden4)
        self.fc5 = nn.Linear(n_hidden4, n_output)
        # 初始化网络权重时，使用了Xavier正态分布初始化方法；
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc5.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DeepQNetwork:
    def __init__(self, n_actions, n_features, n_hidden1=6, n_hidden2=12, n_hidden3=12, n_hidden4=64, n_hidden5=64,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=0.001,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.n_hidden4 = n_hidden4
        self.n_hidden5 = n_hidden5

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.q_values = {}
        # total learning step
        self.learn_step_counter = 0
        self.actions_value_cl = []

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.loss_func = nn.MSELoss()
        self.cost_his = []
        self.epsilon_arr = []
        self._build_net()

    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden1, self.n_hidden2, self.n_hidden3, self.n_hidden4,
                          self.n_actions)
        self.q_target = Net(self.n_features, self.n_hidden1, self.n_hidden2, self.n_hidden3, self.n_hidden4,
                            self.n_actions)
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory 当index超过memory_size之后，置零
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # 先处理成二维数据，再转换为Tensor类型
        observation = torch.Tensor(observation[np.newaxis, :])
        # ε-贪婪法，引入一定随机概率，防止陷入局部最优
        self.epsilon_arr += [self.epsilon]
        if np.random.uniform() < self.epsilon:
            # 通过NN获得action_value,即动作价值函数,1x8的Tensor数组
            actions_value = self.q_eval(observation)
            a = 0
            if a == 0:
                a = observation
            if observation.equal(a):
                self.actions_value_cl.append([actions_value[0][1], actions_value[0][2], actions_value[0][5]])
            # 选择最大动作价值对应的action
            action = np.argmax(actions_value.data.numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return action, self.actions_value_cl

    def learn(self):
        # check to replace target parameters
        # 每学习replace_target_iter（200）次,更新网络权值
        if self.learn_step_counter % self.replace_target_iter == 0:
            # 将预训练的参数权重加载到新的模型之中
            # 此处即时将q_eval网络的参数赋给q_target网络
            # state_dict()返回网络的权值(weight)和偏置(bias)
            self.q_target.load_state_dict(self.q_eval.state_dict())
            print("\ntarget params replaced\n")

        # sample batch memory from all memory
        # 随机选取一个分支的memory，用于学习，更新q_eval的参数
        # 如果大于最大缓存memory_size，则从memory_size的空间中随机选择
        # 如果小于（未存满缓存），则从当前的缓存大小memory_counter中随机选择
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # q_next is used for getting which action would be choosed by target network in state s_(t+1)
        # 对batch_memory取不同的索引值，选择state和state_
        q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(
            torch.Tensor(batch_memory[:, :self.n_features]))
        # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has
        # not been selected unchanged, so when we do q_target - q_eval, these Q_value become zero and wouldn't affect
        # the calculation of the loss 将q_eval复制一份到q_target
        q_target = torch.Tensor(q_eval.data.numpy().copy())
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = torch.Tensor(batch_memory[:, self.n_features + 1])
        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]
        # 计算损失
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increase epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return loss.detach()

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    # Plotting the results for the number of steps
    def plot_results(self, steps, cost, flag_step, fail_times, action_value):

        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('实现done状态的次数', fontproperties=font)
        ax1.set_ylabel('步数', fontproperties=font)
        ax1.set_title('迭代次数与步数', fontproperties=font)

        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('更新网络参数的次数', fontproperties=font)
        ax2.set_ylabel('Loss')
        ax2.set_title('神经网络训练损失', fontproperties=font)

        plt.tight_layout()  # Function to make distance between figures

        fz = plt.figure()
        plt.plot(np.arange(len(flag_step)), flag_step)
        plt.xlabel('到达终点的次数', fontproperties=font)
        plt.ylabel('步数', fontproperties=font)
        plt.title('到达终点步数进化曲线', fontproperties=font)

        f2 = plt.figure()
        plt.plot(np.arange(len(fail_times)), fail_times)
        plt.xlabel('障碍物序号', fontproperties=font)
        plt.ylabel('碰撞次数', fontproperties=font)
        plt.title('各障碍物的累计碰撞次数', fontproperties=font)

        action_value = np.array(action_value)  # 将action_value数组转换为nparray
        f3 = plt.figure()
        plt.plot(np.arange(len(action_value)), action_value)
        plt.xlabel('在起始点选择动作（重置）的次数', fontproperties=font)
        plt.ylabel('动作价值', fontproperties=font)
        plt.legend(['down', 'right', 'right-down'])
        plt.title('起始点的部分动作价值比较', fontproperties=font)

        # 找到每一行的最大值所在的列
        max_cols = np.argmax(action_value, axis=1)
        # 统计每一列出现的次数
        counts = np.bincount(max_cols, minlength=3)
        # 绘制直方图
        f3 = plt.figure()
        plt.bar(range(len(counts)), counts, width=0.3)
        plt.xticks(range(len(counts)), ['down', 'right', 'right-down'])
        plt.ylabel('总数', fontproperties=font)
        plt.title('起始点部分动作被选择的次数', fontproperties=font)

        # Showing the plots
        plt.show()
