from env3 import Environment
from RL_brain_complex2 import DeepQNetwork
from torchsummary import summary
import winsound


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
            if (total_step > 8192) and (total_step % 4 == 0):
                loss = RL.learn()
                loss_rec += [loss]
            observation = observation_
            if done:
                step_rec += [step]  # 只有在到达终点时，才会记录步数
                break
            if step > 1500:
                break
            # print(total_step)
            step += 1
            total_step += 1
            # print(total_step)

    print('flag_times=%d' % flag_times)
    print(env.ob_index)
    # print(RL.epsilon_arr)
    print('game over')
    env.final()
    winsound.Beep(440, 500)
    winsound.Beep(440, 500)
    winsound.Beep(440, 500)

    RL.plot_results(step_rec, loss_rec, env.ob_index)


if __name__ == '__main__':
    env = Environment()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      n_hidden1=625,
                      n_hidden2=24,
                      n_hidden3=24,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=4096,
                      batch_size=64
                      )
    env.after(100, run_maze)
    env.mainloop()


