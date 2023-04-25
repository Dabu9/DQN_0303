from env_p import Environment
# from RL_brain_complex2 import DeepQNetwork
from RL_brain_25x25 import DeepQNetwork
import time

from torchsummary import summary
import winsound


def run_maze():
    step_rec = []  # 记录每次迭代的步数
    flag_step_rec = []  # 记录成功的步数变化
    loss_rec = []  # 记录学习过程中的loss
    total_step = 0  # 记录完整步数
    flag_times = 0
    for episode in range(8000):
        step = 0
        loss = 0
        print("episode: {}".format(episode))
        observation = env.reset()
        while True:
            # print("step: {}".format(step))
            env.render()
            action, action_value = RL.choose_action(observation)
            observation_, reward, done, flag_times = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            # 前200次用于建立数据集
            if (total_step > 1200) and (total_step % 5 == 0):
                loss = RL.learn()
                loss_rec += [loss]
            observation = observation_
            if done:
                if reward == 1:
                    flag_step_rec += [step]
                step_rec += [step]  # 只有在到达终点时，才会记录步数
                break
            if step > 500:
                break
            # print(total_step)
            step += 1
            total_step += 1
            # print(total_step)

    print('flag_times=%d' % flag_times)
    print(env.obstacles_index)
    # print(RL.epsilon_arr)
    print('game over')
    env.final()
    winsound.Beep(440, 500)
    winsound.Beep(440, 500)
    winsound.Beep(440, 500)

    RL.plot_results(step_rec, loss_rec, flag_step_rec, env.obstacles_index, action_value)


if __name__ == '__main__':
    env = Environment()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      n_hidden1=512,
                      n_hidden2=256,
                      n_hidden3=128,
                      n_hidden4=128,
                      n_hidden5=64,
                      learning_rate=0.008,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=100,
                      memory_size=1500,
                      batch_size=32
                      )
    env.after(100, run_maze)
    env.mainloop()


