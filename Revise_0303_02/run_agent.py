# Importing classes
from env import Environment
from agent_brain import DeepQNetwork


def update():
    # 步数
    steps = []

    # 累计损失函数
    all_costs = []

    for episode in range(6000):
        # 初始化Observation
        observation = env.reset()

        # 每一个Episode的Step
        i = 0

        # 每一个Episode的损失函数
        cost = 0

        while True:
            # 更新环境
            env.render()

            # 选择动作
            action = RL.choose_action(observation)

            # 执行动作，并返回observation和reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            # 计算cost
            # cost += RL.learn(observation, action, reward, observation_)
            if (i > 500) and (i % 5 == 0):
                RL.learn()
            # 迭代observation
            observation = observation_

            # 步数计数
            i += 1

            # 当触碰到障碍物或到达终点 结束本次episode
            if done:
                steps += [i]
                all_costs += [cost]
                break

    # 显示最终路径
    env.final()

    # 打印Q_table
    # RL.print_q_table()

    # 结果绘制
    RL.plot_results(steps, all_costs)


if __name__ == "__main__":
    # 创建地图环境
    env = Environment()
    # 构建主算法
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000
                      )
    # 通过调用update函数运行主循环
    env.after(100, update)  # Or just update()
    env.mainloop()
