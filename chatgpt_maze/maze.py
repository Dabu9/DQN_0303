import tkinter as tk
import numpy as np
import time

EPISODES = 1000
# 迷宫大小
MAZE_H = 6
MAZE_W = 9

# 迷宫墙的位置
WALLS = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 5), (4, 6), (4, 7), (1, 8), (2, 8), (3, 8)]

# 起点和终点的位置
START = (2, 0)
GOAL = (0, 8)

# 智能体的初始位置
AGENT_INIT_POS = START

# 学习速率
ALPHA = 0.1

# 探索率
EPSILON = 0.1

# 折扣率
GAMMA = 0.9


class Maze(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("迷宫强化学习")
        self.geometry('{0}x{1}'.format(MAZE_W*50, MAZE_H*50))
        self.action_space = ['上', '下', '左', '右']
        self.n_actions = len(self.action_space)
        self.build_maze()
        self.q_table = np.zeros((MAZE_H*MAZE_W, self.n_actions))

    def build_maze(self):
        # 创建画布
        self.canvas = tk.Canvas(self, bg='white', width=MAZE_W*50, height=MAZE_H*50)

        # 画出迷宫墙
        for i in range(MAZE_H):
            for j in range(MAZE_W):
                if (i, j) in WALLS:
                    self.canvas.create_rectangle(j*50, i*50, j*50+50, i*50+50, fill='black', width=0)
                else:
                    self.canvas.create_rectangle(j*50, i*50, j*50+50, i*50+50, fill='white', width=1)

        # 画出起点和终点
        self.agent = self.canvas.create_oval(AGENT_INIT_POS[1]*50+10, AGENT_INIT_POS[0]*50+10,
                                             AGENT_INIT_POS[1]*50+40, AGENT_INIT_POS[0]*50+40, fill='red')
        self.goal = self.canvas.create_oval(GOAL[1]*50+10, GOAL[0]*50+10,
                                            GOAL[1]*50+40, GOAL[0]*50+40, fill='green')

        # 放置画布
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_oval(AGENT_INIT_POS[1]*50+10, AGENT_INIT_POS[0]*50+10,
                                             AGENT_INIT_POS[1]*50+40, AGENT_INIT_POS[0]*50+40, fill='red')
        return self.canvas.coords(self.agent)

    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:  # 上
            if s[1] > 50:
                base_action[1] -= 50
        elif action == 1:  # 下
            if s[1] < (MAZE_H - 1) * 50:
                base_action[1] += 50
        elif action == 2:  # 左
            if s[0] > 0:
                base_action[0] -= 50
        elif action == 3:  # 右
            if s[0] < (MAZE_W - 1) * 50:
                base_action[0] += 50
        self.canvas.move(self.agent, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.agent)
        if s_ == self.canvas.coords(self.goal):
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        return s_, reward, done

    def render(self):
        time.sleep(0.05)
        self.update()

    def get_state_index(self, state):
        x = int((state[0] - 10) / 50)
        y = int((state[1] - 10) / 50)
        return y * MAZE_W + x

    def choose_action(self, state):
        state_index = self.get_state_index(state)
        if np.random.uniform() < EPSILON:
            action_index = np.random.randint(0, self.n_actions)
        else:
            q_values = self.q_table[state_index, :]
            action_index = np.argmax(q_values)
        return action_index

    def learn(self, s, a, r, s_, done):
        s_index = self.get_state_index(s)
        s_index_ = self.get_state_index(s_)
        if done:
            self.q_table[s_index, a] += ALPHA * (r - self.q_table[s_index, a])
        else:
            self.q_table[s_index, a] += ALPHA * (
                        r + GAMMA * np.max(self.q_table[s_index_, :]) - self.q_table[s_index, a])

if __name__ == '__main__':
    env = Maze()
    q_table = np.zeros((MAZE_H * MAZE_W, 4))
    for i_episode in range(EPISODES):
        state = env.reset()
        while True:
            action = np.argmax(q_table[env.get_state_index(state), :])
            state_, reward, done = env.step(action)
            q_table[env.get_state_index(state), action] += ALPHA * (reward + GAMMA * np.max(q_table[env.get_state_index(state_), :]) - q_table[env.get_state_index(state), action])
            state = state_
            if done:
                break
    env.mainloop()