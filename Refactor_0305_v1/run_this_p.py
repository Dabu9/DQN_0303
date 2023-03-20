from maze_env_private import Maze_pravite
from RL_brain import DeepQNetwork
from torchsummary import summary


def run_maze():
    loss = []
    step_rec = []
    for episode in range(1000):
        # print("episode: {}".format(episode))
        observation = env.reset()
        print(episode)
        step = 0
        while True:
            # print("step: {}".format(step))
            env.render(delay=0)
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            # 前200次用于建立数据集,不训练，在200次后每10次训练模型
            if (step > 200) and (step % 10 == 0):
                loss += [RL.learn()]
            observation = observation_
            if done or step > 6000:
                step_rec += [step]
                break
            step += 1
    # 显示最终路径
    env.final()
    RL.plot_results(step_rec, loss)


if __name__ == '__main__':
    env = Maze_pravite()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=3000
                      )
    env.after(100, run_maze)
    env.mainloop()
