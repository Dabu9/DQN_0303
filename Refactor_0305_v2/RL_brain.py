import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import copy

np.random.seed(1)
torch.manual_seed(1)


# 定义网络结构
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # nn.Linear(in_feature, out_feature)  全连接层
        # n_feature 指observation的size, n_hidden = 20  n_output = n_action
        self.el = nn.Linear(n_feature, n_hidden)   # evaluate net
        self.q = nn.Linear(n_hidden, n_output)     # q_target net

    def forward(self, x):
        x = self.el(x)
        x = F.relu(x)
        x = self.q(x)
        return x


class DeepQNetwork:
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.9 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.loss_func = nn.MSELoss()
        self.cost_his = []

        self._build_net()

    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
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
        if np.random.uniform() < self.epsilon:
            # 通过NN获得action_value,即动作价值函数,1x4的Tensor数组
            actions_value = self.q_eval(observation)
            # 选择最大动作价值对应的action
            action = np.argmax(actions_value.data.numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        # 每学习replace_target_iter（200）次
        if self.learn_step_counter % self.replace_target_iter == 0:
            # 将预训练的参数权重加载到新的模型之中
            # 此处即时将q_eval网络的参数赋给q_target网络
            # state_dict()返回网络的权值(weight)和偏置(bias)
            self.q_target.load_state_dict(self.q_eval.state_dict())
            # print("\ntarget params replaced\n")

        # sample batch memory from all memory
        # 随机选取一个分支的memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # q_next is used for getting which action would be choosed by target network in state s_(t+1)
        # 对batch_memory取不同的索引值，选择state和state_
        q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(
            torch.Tensor(batch_memory[:, :self.n_features]))
        # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
        # so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
        # 将q_eval复制一份到q_target
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
        self.cost_his.append(loss.detach())
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return loss.detach()

# Plotting the results for the number of steps
    def plot_results(self, steps, cost):
        #
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        #
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        plt.tight_layout()  # Function to make distance between figures

        # Showing the plots
        plt.show()