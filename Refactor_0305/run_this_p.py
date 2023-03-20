from maze_env_private import Maze_pravite
from RL_brain import DeepQNetwork
from torchsummary import summary


def run_maze():
    loss = []
    step = 0
    step_rec = []
    for episode in range(2800):
        # print("episode: {}".format(episode))
        observation = env.reset()
        print(episode)
        while True:
            # print("step: {}".format(step))
            env.render(delay=0)
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            # 前200次用于建立数据集
            if (step > 200) and (step % 5 == 0):
                loss += [RL.learn()]
            observation = observation_
            if done:
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
                      memory_size=2000
                      )
    env.after(100, run_maze)
    env.mainloop()
