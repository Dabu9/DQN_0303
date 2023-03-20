import numpy as np
import time
import sys
import itertools
import random
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
# maze_data = [6,   5,   6]
# map_data [seed,size,hell]
# maze_data = [6,   5,   6]
maze_data = [8,   10,   30]

random.seed(maze_data[0])
UNIT = 20   # 单元格大小
padding = 2
block_size = UNIT/2-padding
MAZE_H = maze_data[1]  # grid height
MAZE_W = maze_data[1]  # grid width
hell_size = maze_data[2]


class Maze_pravite(tk.Tk, object):
    def __init__(self):  # 窗口初始化
        super(Maze_pravite, self).__init__()
        self.attributes("-topmost", 1)
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 64
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

        # Dictionaries to draw the final route
        self.d = {}
        self.f = {}

        # Key for the dictionaries
        self.i = 0

        # Writing the final dictionary first time
        self.c = True

        # Showing the steps for longest found route
        self.longest = 0

        # Showing the steps for the shortest route
        self.shortest = 0

    def _build_maze(self):  # 创建初始画布
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([UNIT/2, UNIT/2])
        # 首先通过itertools生成随机坐标,选取hell_size+2个中心坐标点，其中的2就是包括了起始点和终点的坐标
        self.rand = origin[0] + np.array(random.sample(list(itertools.product(range(0, MAZE_H),range(0, MAZE_W))),
                                                       hell_size+2)) * UNIT
        # rand = origin[0] + np.random.randint(1, MAZE_H, size=(1,2))
        # rand_x = origin[0] + np.random.randint(1, MAZE_H, size=hell_size+1) * UNIT
        # rand_y = origin[1] + np.random.randint(1, MAZE_W, size=hell_size+1) * UNIT
        self.hell = {}
        # print(self.rand)
        for t in range(hell_size):
            # print(t)
            self.hell[t] = self.canvas.create_rectangle(
                self.rand[t][0] - block_size, self.rand[t][1] - block_size,
                self.rand[t][0] + block_size, self.rand[t][1] + block_size,
                fill='black')

        # create oval
        oval_center = origin + UNIT * 3
        self.oval = self.canvas.create_oval(
            self.rand[hell_size][0] - block_size, self.rand[hell_size][1] - block_size,
            self.rand[hell_size][0] + block_size, self.rand[hell_size][1] + block_size,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            self.rand[hell_size-1][0] - block_size, self.rand[hell_size-1][0] - block_size,
            self.rand[hell_size-1][0] + block_size, self.rand[hell_size-1][0] + block_size,
            fill='red')

        # pack all
        self.canvas.pack()
        self.cors = []
        for s in range(hell_size):
            self.cors.append(self.canvas.coords(self.hell[s]))

    def reset(self):    # 初始化智能体agent
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect)  # 画布清除
        origin = np.array([20, 20])
        # create_rectangle(x0, y0, x1, y1, options)
        self.canvas.create_rectangle(
            self.rand[hell_size - 1][0] - block_size, self.rand[hell_size - 1][0] - block_size,
            self.rand[hell_size - 1][0] + block_size, self.rand[hell_size - 1][0] + block_size,
            fill='blue')
        self.rect = self.canvas.create_rectangle(
            self.rand[hell_size - 1][0] - block_size, self.rand[hell_size - 1][0] - block_size,
            self.rand[hell_size - 1][0] + block_size, self.rand[hell_size - 1][0] + block_size,
            fill='red')

        # return observation
        agent_pos = np.array(self.canvas.coords(self.rect)[:2])  # 智能体方块的位置
        target_pos = np.array(self.canvas.coords(self.oval)[:2])  # 目标方块的位置
        obstacle_pos_list = []  # 存储障碍物位置信息的列表
        for t in range(hell_size):
            obstacle_pos = np.array(self.canvas.coords(self.hell[t])[:2])
            obstacle_pos_list.append(obstacle_pos)
        state = np.concatenate((agent_pos, target_pos, *obstacle_pos_list)) / (
                    MAZE_W * UNIT)  # 将代理方块、目标方块和障碍物的位置信息拼接成一个向量，并进行归一化

        return state

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        # print(base_action)
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state
        # reward function
        if next_coords == self.canvas.coords(self.oval):  # 到达终点
            reward = 1
            done = True
        elif next_coords in self.cors:       #
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        agent_pos = np.array(self.canvas.coords(self.rect)[:2])  # 智能体方块的位置
        target_pos = np.array(self.canvas.coords(self.oval)[:2])  # 目标方块的位置
        obstacle_pos_list = []  # 存储障碍物位置信息的列表
        for t in range(hell_size):
            obstacle_pos = np.array(self.canvas.coords(self.hell[t])[:2])
            obstacle_pos_list.append(obstacle_pos)
        s_ = np.concatenate((agent_pos, target_pos, *obstacle_pos_list)) / (
                    MAZE_W * UNIT)  # 将代理方块、目标方块和障碍物的位置信息拼接成一个向量，并进行归一化

        return s_, reward, done

    def render(self, delay=0.1):
        # time.sleep(delay)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze_pravite()
    env.after(100, update)  # 每1s执行一次update()函数
    env.mainloop()
