from env3 import Environment
from RL_brain_complex2 import DeepQNetwork
from torchsummary import summary


def run_maze():
    step_rec = []  # 记录每次迭代的步数
    loss_rec = []  # 记录学习过程中的loss
    total_step = 0  # 记录完整步数
    flag_times = 0
    for episode in range(10000):
        step = 0
        loss = 0
        print("episode: {}".format(episode))
        observation = env.reset()
        while True:
            # print("step: {}".format(step))
            env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, flag_times = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            # 前200次用于建立数据集
            if (total_step > 2000) and (total_step % 5 == 0):
                loss = RL.learn()
                loss_rec += [loss]
            observation = observation_
            if done:
                step_rec += [step]  # 只有在到达终点时，才会记录步数
                break

            step += 1
            total_step += 1
            # print(total_step)

    print('flag_times=%d' % flag_times)
    print('game over')
    env.final()
    RL.plot_results(step_rec, loss_rec)


if __name__ == '__main__':
    env = Environment()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      n_hidden1=12,
                      n_hidden2=24,
                      n_hidden3=4,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=6000
                      )
    env.after(100, run_maze)
    env.mainloop()

# 有几个可能的改进方法：
#
# 调整学习率（learning rate）：如果学习率设置过高，可能会导致训练不收敛。可以尝试减小学习率，例如将learning_rate调整为0.001或更小的值。
#
# 调整epsilon值：epsilon值控制着探索（exploration）和利用（exploitation）之间的平衡。可以逐步减小epsilon值，让模型在训练后期更加倾向于利用已经学习到的经验，例如将epsilon_max值从0.9减小到0.1或更小的值。
#
# 调整神经网络架构：可能存在过拟合或欠拟合的问题，可以调整神经网络的结构，例如增加隐藏层节点数或减少隐藏层层数，尝试更好的拟合数据。
#
# 使用经验回放（experience replay）：经验回放可以帮助缓解数据相关性带来的影响，使得训练更加稳定。在训练过程中，可以将经验存储在一个经验池中，然后随机采样经验进行训练。
#
# 调整reward的计算方式：reward值的计算方式可能会影响训练效果。可以尝试不同的reward计算方式，例如使用稀疏的reward或者使用更丰富的reward来指导模型学习。
#
# 以上是几个可能的改进方法，具体的改进方法需要根据具体情况进行选择。

