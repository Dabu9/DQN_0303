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
# map_data.py [seed,size,hell]
# maze_data = [6,   5,   6]
maze_data = [2, 25, 24]

random.seed(maze_data[0])
UNIT = 20  # 单元格大小
padding = 2
block_size = UNIT / 2 - padding
MAZE_H = maze_data[1]  # grid height
MAZE_W = maze_data[1]  # grid width
hell_size = maze_data[2]
# Global variable for dictionary with coordinates for the final route
a = {}
start_loc = [22, 6]  # 起点坐标定义（x,y），0表示第一列/行，注意不要与障碍物坐标重复
end_loc = [19, 19]  # 终点坐标定义

class Maze_pravite(tk.Tk, object):
    def __init__(self):  # 窗口初始化
        super(Maze_pravite, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
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
        self.origin = np.array([UNIT / 2, UNIT / 2])
        # 首先通过itertools生成随机坐标,选取hell_size+2个中心坐标点，其中的2就是包括了起始点和终点的坐标
        self.rand = self.origin[0] + np.array(random.sample(list(itertools.product(range(0, MAZE_H), range(0, MAZE_W))),
                                                            hell_size + 2)) * UNIT
        # rand = origin[0] + np.random.randint(1, MAZE_H, size=(1,2))
        # rand_x = origin[0] + np.random.randint(1, MAZE_H, size=hell_size+1) * UNIT
        # rand_y = origin[1] + np.random.randint(1, MAZE_W, size=hell_size+1) * UNIT
        # self.hell = {}

        # Obstacle 1
        # Defining the center of obstacle 1
        obstacle1_center = self.origin + np.array([UNIT, UNIT * 2])
        # Building the obstacle 1
        self.obstacle1 = self.canvas.create_rectangle(
            obstacle1_center[0] - block_size, obstacle1_center[1] - block_size,  # Top left corner
            obstacle1_center[0] + block_size, obstacle1_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 1 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle1 = [self.canvas.coords(self.obstacle1)[0] + 3,
                                 self.canvas.coords(self.obstacle1)[1] + 3,
                                 self.canvas.coords(self.obstacle1)[2] - 3,
                                 self.canvas.coords(self.obstacle1)[3] - 3]

        # Obstacle 2
        # Defining the center of obstacle 2
        obstacle2_center = self.origin + np.array([UNIT * 2, UNIT * 2])
        # Building the obstacle 2
        self.obstacle2 = self.canvas.create_rectangle(
            obstacle2_center[0] - block_size, obstacle2_center[1] - block_size,  # Top left corner
            obstacle2_center[0] + block_size, obstacle2_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 2 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle2 = [self.canvas.coords(self.obstacle2)[0] + 3,
                                 self.canvas.coords(self.obstacle2)[1] + 3,
                                 self.canvas.coords(self.obstacle2)[2] - 3,
                                 self.canvas.coords(self.obstacle2)[3] - 3]

        # Obstacle 3
        # Defining the center of obstacle 3
        obstacle3_center = self.origin + np.array([UNIT * 3, UNIT * 2])
        # Building the obstacle 3
        self.obstacle3 = self.canvas.create_rectangle(
            obstacle3_center[0] - block_size, obstacle3_center[1] - block_size,  # Top left corner
            obstacle3_center[0] + block_size, obstacle3_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 3 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle3 = [self.canvas.coords(self.obstacle3)[0] + 3,
                                 self.canvas.coords(self.obstacle3)[1] + 3,
                                 self.canvas.coords(self.obstacle3)[2] - 3,
                                 self.canvas.coords(self.obstacle3)[3] - 3]

        # Obstacle 4
        # Defining the center of obstacle 4
        obstacle4_center = self.origin + np.array([UNIT * 3, UNIT * 3])
        # Building the obstacle 4
        self.obstacle4 = self.canvas.create_rectangle(
            obstacle4_center[0] - block_size, obstacle4_center[1] - block_size,  # Top left corner
            obstacle4_center[0] + block_size, obstacle4_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 4 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle4 = [self.canvas.coords(self.obstacle4)[0] + 3,
                                 self.canvas.coords(self.obstacle4)[1] + 3,
                                 self.canvas.coords(self.obstacle4)[2] - 3,
                                 self.canvas.coords(self.obstacle4)[3] - 3]

        # Obstacle 5
        # Defining the center of obstacle 5
        obstacle5_center = self.origin + np.array([UNIT * 4, UNIT * 10])
        # Building the obstacle 5
        self.obstacle5 = self.canvas.create_rectangle(
            obstacle5_center[0] - block_size, obstacle5_center[1] - block_size,  # Top left corner
            obstacle5_center[0] + block_size, obstacle5_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 2 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle5 = [self.canvas.coords(self.obstacle5)[0] + 3,
                                 self.canvas.coords(self.obstacle5)[1] + 3,
                                 self.canvas.coords(self.obstacle5)[2] - 3,
                                 self.canvas.coords(self.obstacle5)[3] - 3]

        # Obstacle 6
        # Defining the center of obstacle 6
        obstacle6_center = self.origin + np.array([UNIT * 4, UNIT * 11])
        # Building the obstacle 6
        self.obstacle6 = self.canvas.create_rectangle(
            obstacle6_center[0] - block_size, obstacle6_center[1] - block_size,  # Top left corner
            obstacle6_center[0] + block_size, obstacle6_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 6 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle6 = [self.canvas.coords(self.obstacle6)[0] + 3,
                                 self.canvas.coords(self.obstacle6)[1] + 3,
                                 self.canvas.coords(self.obstacle6)[2] - 3,
                                 self.canvas.coords(self.obstacle6)[3] - 3]

        # Obstacle 7
        # Defining the center of obstacle 7
        obstacle7_center = self.origin + np.array([UNIT * 4, UNIT * 12])
        # Building the obstacle 7
        self.obstacle7 = self.canvas.create_rectangle(
            obstacle7_center[0] - block_size, obstacle7_center[1] - block_size,  # Top left corner
            obstacle7_center[0] + block_size, obstacle7_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 7 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle7 = [self.canvas.coords(self.obstacle7)[0] + 3,
                                 self.canvas.coords(self.obstacle7)[1] + 3,
                                 self.canvas.coords(self.obstacle7)[2] - 3,
                                 self.canvas.coords(self.obstacle7)[3] - 3]

        # Obstacle 8
        # Defining the center of obstacle 8
        obstacle8_center = self.origin + np.array([UNIT * 5, UNIT * 12])
        # Building the obstacle 8
        self.obstacle8 = self.canvas.create_rectangle(
            obstacle8_center[0] - block_size, obstacle8_center[1] - block_size,  # Top left corner
            obstacle8_center[0] + block_size, obstacle8_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 8 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle8 = [self.canvas.coords(self.obstacle8)[0] + 3,
                                 self.canvas.coords(self.obstacle8)[1] + 3,
                                 self.canvas.coords(self.obstacle8)[2] - 3,
                                 self.canvas.coords(self.obstacle8)[3] - 3]

        # Obstacle 9
        # Defining the center of obstacle 9
        obstacle9_center = self.origin + np.array([UNIT * 6, UNIT * 12])
        # Building the obstacle 9
        self.obstacle9 = self.canvas.create_rectangle(
            obstacle9_center[0] - block_size, obstacle9_center[1] - block_size,  # Top left corner
            obstacle9_center[0] + block_size, obstacle9_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 9 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle9 = [self.canvas.coords(self.obstacle9)[0] + 3,
                                 self.canvas.coords(self.obstacle9)[1] + 3,
                                 self.canvas.coords(self.obstacle9)[2] - 3,
                                 self.canvas.coords(self.obstacle9)[3] - 3]

        # Obstacle 10
        # Defining the center of obstacle 10
        obstacle10_center = self.origin + np.array([UNIT * 2, UNIT * 18])
        # Building the obstacle 10
        self.obstacle10 = self.canvas.create_rectangle(
            obstacle10_center[0] - block_size, obstacle10_center[1] - block_size,  # Top left corner
            obstacle10_center[0] + block_size, obstacle10_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 10 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle10 = [self.canvas.coords(self.obstacle10)[0] + 3,
                                  self.canvas.coords(self.obstacle10)[1] + 3,
                                  self.canvas.coords(self.obstacle10)[2] - 3,
                                  self.canvas.coords(self.obstacle10)[3] - 3]

        # Obstacle 11
        # Defining the center of obstacle 11
        obstacle11_center = self.origin + np.array([UNIT * 3, UNIT * 18])
        # Building the obstacle 11
        self.obstacle11 = self.canvas.create_rectangle(
            obstacle11_center[0] - block_size, obstacle11_center[1] - block_size,  # Top left corner
            obstacle11_center[0] + block_size, obstacle11_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 11 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle11 = [self.canvas.coords(self.obstacle11)[0] + 3,
                                  self.canvas.coords(self.obstacle11)[1] + 3,
                                  self.canvas.coords(self.obstacle11)[2] - 3,
                                  self.canvas.coords(self.obstacle11)[3] - 3]

        # Obstacle 12
        # Defining the center of obstacle 12
        obstacle12_center = self.origin + np.array([UNIT * 4, UNIT * 18])
        # Building the obstacle 12
        self.obstacle12 = self.canvas.create_rectangle(
            obstacle12_center[0] - block_size, obstacle12_center[1] - block_size,  # Top left corner
            obstacle12_center[0] + block_size, obstacle12_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 12 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle12 = [self.canvas.coords(self.obstacle12)[0] + 3,
                                  self.canvas.coords(self.obstacle12)[1] + 3,
                                  self.canvas.coords(self.obstacle12)[2] - 3,
                                  self.canvas.coords(self.obstacle12)[3] - 3]

        # Obstacle 13
        # Defining the center of obstacle 13
        obstacle13_center = self.origin + np.array([UNIT * 3, UNIT * 19])
        # Building the obstacle 13
        self.obstacle13 = self.canvas.create_rectangle(
            obstacle13_center[0] - block_size, obstacle13_center[1] - block_size,  # Top left corner
            obstacle13_center[0] + block_size, obstacle13_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 13 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle13 = [self.canvas.coords(self.obstacle13)[0] + 3,
                                  self.canvas.coords(self.obstacle13)[1] + 3,
                                  self.canvas.coords(self.obstacle13)[2] - 3,
                                  self.canvas.coords(self.obstacle13)[3] - 3]

        # Obstacle 14
        # Defining the center of obstacle 14
        obstacle14_center = self.origin + np.array([UNIT * 3, UNIT * 20])
        # Building the obstacle 14
        self.obstacle14 = self.canvas.create_rectangle(
            obstacle14_center[0] - block_size, obstacle14_center[1] - block_size,  # Top left corner
            obstacle14_center[0] + block_size, obstacle14_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 14 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle14 = [self.canvas.coords(self.obstacle14)[0] + 3,
                                  self.canvas.coords(self.obstacle14)[1] + 3,
                                  self.canvas.coords(self.obstacle14)[2] - 3,
                                  self.canvas.coords(self.obstacle14)[3] - 3]

        # Obstacle 15
        # Defining the center of obstacle 15
        obstacle15_center = self.origin + np.array([UNIT * 3, UNIT * 21])
        # Building the obstacle 15
        self.obstacle15 = self.canvas.create_rectangle(
            obstacle15_center[0] - block_size, obstacle15_center[1] - block_size,  # Top left corner
            obstacle15_center[0] + block_size, obstacle15_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 15 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle15 = [self.canvas.coords(self.obstacle15)[0] + 3,
                                  self.canvas.coords(self.obstacle15)[1] + 3,
                                  self.canvas.coords(self.obstacle15)[2] - 3,
                                  self.canvas.coords(self.obstacle15)[3] - 3]

        # Obstacle 16
        # Defining the center of obstacle 16
        obstacle16_center = self.origin + np.array([UNIT * 10, UNIT * 22])
        # Building the obstacle 16
        self.obstacle16 = self.canvas.create_rectangle(
            obstacle16_center[0] - block_size, obstacle16_center[1] - block_size,  # Top left corner
            obstacle16_center[0] + block_size, obstacle16_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 16 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle16 = [self.canvas.coords(self.obstacle16)[0] + 3,
                                  self.canvas.coords(self.obstacle16)[1] + 3,
                                  self.canvas.coords(self.obstacle16)[2] - 3,
                                  self.canvas.coords(self.obstacle16)[3] - 3]

        # Obstacle 17
        # Defining the center of obstacle 17
        obstacle17_center = self.origin + np.array([UNIT * 11, UNIT * 15])
        # Building the obstacle 17
        self.obstacle17 = self.canvas.create_rectangle(
            obstacle17_center[0] - block_size, obstacle17_center[1] - block_size,  # Top left corner
            obstacle17_center[0] + block_size, obstacle17_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 17 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle17 = [self.canvas.coords(self.obstacle17)[0] + 3,
                                  self.canvas.coords(self.obstacle17)[1] + 3,
                                  self.canvas.coords(self.obstacle17)[2] - 3,
                                  self.canvas.coords(self.obstacle17)[3] - 3]

        # Obstacle 18
        # Defining the center of obstacle 18
        obstacle18_center = self.origin + np.array([UNIT * 12, UNIT * 15])
        # Building the obstacle 18
        self.obstacle18 = self.canvas.create_rectangle(
            obstacle18_center[0] - block_size, obstacle18_center[1] - block_size,  # Top left corner
            obstacle18_center[0] + block_size, obstacle18_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 18 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle18 = [self.canvas.coords(self.obstacle18)[0] + 3,
                                  self.canvas.coords(self.obstacle18)[1] + 3,
                                  self.canvas.coords(self.obstacle18)[2] - 3,
                                  self.canvas.coords(self.obstacle18)[3] - 3]

        # Obstacle 19
        # Defining the center of obstacle 19
        obstacle19_center = self.origin + np.array([UNIT * 13, UNIT * 15])
        # Building the obstacle 19
        self.obstacle19 = self.canvas.create_rectangle(
            obstacle19_center[0] - block_size, obstacle19_center[1] - block_size,  # Top left corner
            obstacle19_center[0] + block_size, obstacle19_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 19 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle19 = [self.canvas.coords(self.obstacle19)[0] + 3,
                                  self.canvas.coords(self.obstacle19)[1] + 3,
                                  self.canvas.coords(self.obstacle19)[2] - 3,
                                  self.canvas.coords(self.obstacle19)[3] - 3]

        # Obstacle 20
        # Defining the center of obstacle 2
        obstacle20_center = self.origin + np.array([UNIT * 13, UNIT * 14])
        # Building the obstacle 20
        self.obstacle20 = self.canvas.create_rectangle(
            obstacle20_center[0] - block_size, obstacle20_center[1] - block_size,  # Top left corner
            obstacle20_center[0] + block_size, obstacle20_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 20 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle20 = [self.canvas.coords(self.obstacle20)[0] + 3,
                                  self.canvas.coords(self.obstacle20)[1] + 3,
                                  self.canvas.coords(self.obstacle20)[2] - 3,
                                  self.canvas.coords(self.obstacle20)[3] - 3]

        # Obstacle 21
        # Defining the center of obstacle 21
        obstacle21_center = self.origin + np.array([UNIT * 13, UNIT * 13])
        # Building the obstacle 21
        self.obstacle21 = self.canvas.create_rectangle(
            obstacle21_center[0] - block_size, obstacle21_center[1] - block_size,  # Top left corner
            obstacle21_center[0] + block_size, obstacle21_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 21 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle21 = [self.canvas.coords(self.obstacle21)[0] + 3,
                                  self.canvas.coords(self.obstacle21)[1] + 3,
                                  self.canvas.coords(self.obstacle21)[2] - 3,
                                  self.canvas.coords(self.obstacle21)[3] - 3]

        # Obstacle 22
        # Defining the center of obstacle 22
        obstacle22_center = self.origin + np.array([UNIT * 21, UNIT * 22])
        # Building the obstacle 22
        self.obstacle22 = self.canvas.create_rectangle(
            obstacle22_center[0] - block_size, obstacle22_center[1] - block_size,  # Top left corner
            obstacle22_center[0] + block_size, obstacle22_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 22 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle22 = [self.canvas.coords(self.obstacle22)[0] + 3,
                                  self.canvas.coords(self.obstacle22)[1] + 3,
                                  self.canvas.coords(self.obstacle22)[2] - 3,
                                  self.canvas.coords(self.obstacle22)[3] - 3]

        # Obstacle 23
        # Defining the center of obstacle 23
        obstacle23_center = self.origin + np.array([UNIT * 20, UNIT * 22])
        # Building the obstacle 23
        self.obstacle23 = self.canvas.create_rectangle(
            obstacle23_center[0] - block_size, obstacle23_center[1] - block_size,  # Top left corner
            obstacle23_center[0] + block_size, obstacle23_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 23 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle23 = [self.canvas.coords(self.obstacle23)[0] + 3,
                                  self.canvas.coords(self.obstacle23)[1] + 3,
                                  self.canvas.coords(self.obstacle23)[2] - 3,
                                  self.canvas.coords(self.obstacle23)[3] - 3]

        # Obstacle 24
        # Defining the center of obstacle 24
        obstacle24_center = self.origin + np.array([UNIT * 19, UNIT * 22])
        # Building the obstacle 24
        self.obstacle24 = self.canvas.create_rectangle(
            obstacle24_center[0] - block_size, obstacle24_center[1] - block_size,  # Top left corner
            obstacle24_center[0] + block_size, obstacle24_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 24 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle24 = [self.canvas.coords(self.obstacle24)[0] + 3,
                                  self.canvas.coords(self.obstacle24)[1] + 3,
                                  self.canvas.coords(self.obstacle24)[2] - 3,
                                  self.canvas.coords(self.obstacle24)[3] - 3]

        # Obstacle 25
        # Defining the center of obstacle 25
        obstacle25_center = self.origin + np.array([UNIT * 18, UNIT * 22])
        # Building the obstacle 25
        self.obstacle25 = self.canvas.create_rectangle(
            obstacle25_center[0] - block_size, obstacle25_center[1] - block_size,  # Top left corner
            obstacle25_center[0] + block_size, obstacle25_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 25 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle25 = [self.canvas.coords(self.obstacle25)[0] + 3,
                                  self.canvas.coords(self.obstacle25)[1] + 3,
                                  self.canvas.coords(self.obstacle25)[2] - 3,
                                  self.canvas.coords(self.obstacle25)[3] - 3]

        # Obstacle 26
        # Defining the center of obstacle 26
        obstacle26_center = self.origin + np.array([UNIT * 18, UNIT * 21])
        # Building the obstacle 26
        self.obstacle26 = self.canvas.create_rectangle(
            obstacle26_center[0] - block_size, obstacle26_center[1] - block_size,  # Top left corner
            obstacle26_center[0] + block_size, obstacle26_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 26 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle26 = [self.canvas.coords(self.obstacle26)[0] + 3,
                                  self.canvas.coords(self.obstacle26)[1] + 3,
                                  self.canvas.coords(self.obstacle26)[2] - 3,
                                  self.canvas.coords(self.obstacle26)[3] - 3]

        # Obstacle 27
        # Defining the center of obstacle 27
        obstacle27_center = self.origin + np.array([UNIT * 18, UNIT * 20])
        # Building the obstacle 27
        self.obstacle27 = self.canvas.create_rectangle(
            obstacle27_center[0] - block_size, obstacle27_center[1] - block_size,  # Top left corner
            obstacle27_center[0] + block_size, obstacle27_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 27 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle27 = [self.canvas.coords(self.obstacle27)[0] + 3,
                                  self.canvas.coords(self.obstacle27)[1] + 3,
                                  self.canvas.coords(self.obstacle27)[2] - 3,
                                  self.canvas.coords(self.obstacle27)[3] - 3]

        # Obstacle 28
        # Defining the center of obstacle 28
        obstacle28_center = self.origin + np.array([UNIT * 18, UNIT * 19])
        # Building the obstacle 28
        self.obstacle28 = self.canvas.create_rectangle(
            obstacle28_center[0] - block_size, obstacle28_center[1] - block_size,  # Top left corner
            obstacle28_center[0] + block_size, obstacle28_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 28 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle28 = [self.canvas.coords(self.obstacle28)[0] + 3,
                                  self.canvas.coords(self.obstacle28)[1] + 3,
                                  self.canvas.coords(self.obstacle28)[2] - 3,
                                  self.canvas.coords(self.obstacle28)[3] - 3]

        # Obstacle 29
        # Defining the center of obstacle 29
        obstacle29_center = self.origin + np.array([UNIT * 18, UNIT * 18])
        # Building the obstacle 29
        self.obstacle29 = self.canvas.create_rectangle(
            obstacle29_center[0] - block_size, obstacle29_center[1] - block_size,  # Top left corner
            obstacle29_center[0] + block_size, obstacle29_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 29 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle29 = [self.canvas.coords(self.obstacle29)[0] + 3,
                                  self.canvas.coords(self.obstacle29)[1] + 3,
                                  self.canvas.coords(self.obstacle29)[2] - 3,
                                  self.canvas.coords(self.obstacle29)[3] - 3]

        # Obstacle 30
        # Defining the center of obstacle 30
        obstacle30_center = self.origin + np.array([UNIT * 19, UNIT * 18])
        # Building the obstacle 30
        self.obstacle30 = self.canvas.create_rectangle(
            obstacle30_center[0] - block_size, obstacle30_center[1] - block_size,  # Top left corner
            obstacle30_center[0] + block_size, obstacle30_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 30 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle30 = [self.canvas.coords(self.obstacle30)[0] + 3,
                                  self.canvas.coords(self.obstacle30)[1] + 3,
                                  self.canvas.coords(self.obstacle30)[2] - 3,
                                  self.canvas.coords(self.obstacle30)[3] - 3]

        # Obstacle 31
        # Defining the center of obstacle 31
        obstacle31_center = self.origin + np.array([UNIT * 20, UNIT * 18])
        # Building the obstacle 31
        self.obstacle31 = self.canvas.create_rectangle(
            obstacle31_center[0] - block_size, obstacle31_center[1] - block_size,  # Top left corner
            obstacle31_center[0] + block_size, obstacle31_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 31 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle31 = [self.canvas.coords(self.obstacle31)[0] + 3,
                                  self.canvas.coords(self.obstacle31)[1] + 3,
                                  self.canvas.coords(self.obstacle31)[2] - 3,
                                  self.canvas.coords(self.obstacle31)[3] - 3]

        # Obstacle 32
        # Defining the center of obstacle 32
        obstacle32_center = self.origin + np.array([UNIT * 11, UNIT * 6])
        # Building the obstacle 32
        self.obstacle32 = self.canvas.create_rectangle(
            obstacle32_center[0] - block_size, obstacle32_center[1] - block_size,  # Top left corner
            obstacle32_center[0] + block_size, obstacle32_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 32 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle32 = [self.canvas.coords(self.obstacle32)[0] + 3,
                                  self.canvas.coords(self.obstacle32)[1] + 3,
                                  self.canvas.coords(self.obstacle32)[2] - 3,
                                  self.canvas.coords(self.obstacle32)[3] - 3]

        # Obstacle 33
        # Defining the center of obstacle 33
        obstacle33_center = self.origin + np.array([UNIT * 12, UNIT * 6])
        # Building the obstacle 33
        self.obstacle33 = self.canvas.create_rectangle(
            obstacle33_center[0] - block_size, obstacle33_center[1] - block_size,  # Top left corner
            obstacle33_center[0] + block_size, obstacle33_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 33 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle33 = [self.canvas.coords(self.obstacle33)[0] + 3,
                                  self.canvas.coords(self.obstacle33)[1] + 3,
                                  self.canvas.coords(self.obstacle33)[2] - 3,
                                  self.canvas.coords(self.obstacle33)[3] - 3]

        # Obstacle 34
        # Defining the center of obstacle 34
        obstacle34_center = self.origin + np.array([UNIT * 13, UNIT * 6])
        # Building the obstacle 34
        self.obstacle34 = self.canvas.create_rectangle(
            obstacle34_center[0] - block_size, obstacle34_center[1] - block_size,  # Top left corner
            obstacle34_center[0] + block_size, obstacle34_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 34 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle34 = [self.canvas.coords(self.obstacle34)[0] + 3,
                                  self.canvas.coords(self.obstacle34)[1] + 3,
                                  self.canvas.coords(self.obstacle34)[2] - 3,
                                  self.canvas.coords(self.obstacle34)[3] - 3]

        # Obstacle 35
        # Defining the center of obstacle 35
        obstacle35_center = self.origin + np.array([UNIT * 14, UNIT * 6])
        # Building the obstacle 35
        self.obstacle35 = self.canvas.create_rectangle(
            obstacle35_center[0] - block_size, obstacle35_center[1] - block_size,  # Top left corner
            obstacle35_center[0] + block_size, obstacle35_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 35 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle35 = [self.canvas.coords(self.obstacle35)[0] + 3,
                                  self.canvas.coords(self.obstacle35)[1] + 3,
                                  self.canvas.coords(self.obstacle35)[2] - 3,
                                  self.canvas.coords(self.obstacle35)[3] - 3]

        # Obstacle 36
        # Defining the center of obstacle 36
        obstacle36_center = self.origin + np.array([UNIT * 14, UNIT * 7])
        # Building the obstacle 36
        self.obstacle36 = self.canvas.create_rectangle(
            obstacle36_center[0] - block_size, obstacle36_center[1] - block_size,  # Top left corner
            obstacle36_center[0] + block_size, obstacle36_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 36 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle36 = [self.canvas.coords(self.obstacle36)[0] + 3,
                                  self.canvas.coords(self.obstacle36)[1] + 3,
                                  self.canvas.coords(self.obstacle36)[2] - 3,
                                  self.canvas.coords(self.obstacle36)[3] - 3]

        # Obstacle 37
        # Defining the center of obstacle 37
        obstacle37_center = self.origin + np.array([UNIT * 14, UNIT * 5])
        # Building the obstacle 37
        self.obstacle37 = self.canvas.create_rectangle(
            obstacle37_center[0] - block_size, obstacle37_center[1] - block_size,  # Top left corner
            obstacle37_center[0] + block_size, obstacle37_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 37 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle37 = [self.canvas.coords(self.obstacle37)[0] + 3,
                                  self.canvas.coords(self.obstacle37)[1] + 3,
                                  self.canvas.coords(self.obstacle37)[2] - 3,
                                  self.canvas.coords(self.obstacle37)[3] - 3]

        # Obstacle 38
        # Defining the center of obstacle 38
        obstacle38_center = self.origin + np.array([UNIT * 20, UNIT])
        # Building the obstacle 38
        self.obstacle38 = self.canvas.create_rectangle(
            obstacle38_center[0] - block_size, obstacle38_center[1] - block_size,  # Top left corner
            obstacle38_center[0] + block_size, obstacle38_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 38 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle38 = [self.canvas.coords(self.obstacle38)[0] + 3,
                                  self.canvas.coords(self.obstacle38)[1] + 3,
                                  self.canvas.coords(self.obstacle38)[2] - 3,
                                  self.canvas.coords(self.obstacle38)[3] - 3]

        # Obstacle 39
        # Defining the center of obstacle 39
        obstacle39_center = self.origin + np.array([UNIT * 20, UNIT * 2])
        # Building the obstacle 39
        self.obstacle39 = self.canvas.create_rectangle(
            obstacle39_center[0] - block_size, obstacle39_center[1] - block_size,  # Top left corner
            obstacle39_center[0] + block_size, obstacle39_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 39 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle39 = [self.canvas.coords(self.obstacle39)[0] + 3,
                                  self.canvas.coords(self.obstacle39)[1] + 3,
                                  self.canvas.coords(self.obstacle39)[2] - 3,
                                  self.canvas.coords(self.obstacle39)[3] - 3]

        # Obstacle 40
        # Defining the center of obstacle 40
        obstacle40_center = self.origin + np.array([UNIT * 20, UNIT * 3])
        # Building the obstacle 40
        self.obstacle40 = self.canvas.create_rectangle(
            obstacle40_center[0] - block_size, obstacle40_center[1] - block_size,  # Top left corner
            obstacle40_center[0] + block_size, obstacle40_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 40 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle40 = [self.canvas.coords(self.obstacle40)[0] + 3,
                                  self.canvas.coords(self.obstacle40)[1] + 3,
                                  self.canvas.coords(self.obstacle40)[2] - 3,
                                  self.canvas.coords(self.obstacle40)[3] - 3]

        # Obstacle 41
        # Defining the center of obstacle 41
        obstacle41_center = self.origin + np.array([UNIT * 20, UNIT * 4])
        # Building the obstacle 41
        self.obstacle41 = self.canvas.create_rectangle(
            obstacle41_center[0] - block_size, obstacle41_center[1] - block_size,  # Top left corner
            obstacle41_center[0] + block_size, obstacle41_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 41 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle41 = [self.canvas.coords(self.obstacle41)[0] + 3,
                                  self.canvas.coords(self.obstacle41)[1] + 3,
                                  self.canvas.coords(self.obstacle41)[2] - 3,
                                  self.canvas.coords(self.obstacle41)[3] - 3]

        # Obstacle 42
        # Defining the center of obstacle 42
        obstacle42_center = self.origin + np.array([UNIT * 21, UNIT * 4])
        # Building the obstacle 42
        self.obstacle42 = self.canvas.create_rectangle(
            obstacle42_center[0] - block_size, obstacle42_center[1] - block_size,  # Top left corner
            obstacle42_center[0] + block_size, obstacle42_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 42 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle42 = [self.canvas.coords(self.obstacle42)[0] + 3,
                                  self.canvas.coords(self.obstacle42)[1] + 3,
                                  self.canvas.coords(self.obstacle42)[2] - 3,
                                  self.canvas.coords(self.obstacle42)[3] - 3]

        # Obstacle 43
        # Defining the center of obstacle 43
        obstacle43_center = self.origin + np.array([UNIT * 19, UNIT * 4])
        # Building the obstacle 43
        self.obstacle43 = self.canvas.create_rectangle(
            obstacle43_center[0] - block_size, obstacle43_center[1] - block_size,  # Top left corner
            obstacle43_center[0] + block_size, obstacle43_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 43 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle43 = [self.canvas.coords(self.obstacle43)[0] + 3,
                                  self.canvas.coords(self.obstacle43)[1] + 3,
                                  self.canvas.coords(self.obstacle43)[2] - 3,
                                  self.canvas.coords(self.obstacle43)[3] - 3]

        # Obstacle 44
        # Defining the center of obstacle 44
        obstacle44_center = self.origin + np.array([UNIT * 17, UNIT * 10])
        # Building the obstacle 44
        self.obstacle44 = self.canvas.create_rectangle(
            obstacle44_center[0] - block_size, obstacle44_center[1] - block_size,  # Top left corner
            obstacle44_center[0] + block_size, obstacle44_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 44 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle44 = [self.canvas.coords(self.obstacle44)[0] + 3,
                                  self.canvas.coords(self.obstacle44)[1] + 3,
                                  self.canvas.coords(self.obstacle44)[2] - 3,
                                  self.canvas.coords(self.obstacle44)[3] - 3]

        # Obstacle 45
        # Defining the center of obstacle 45
        obstacle45_center = self.origin + np.array([UNIT * 18, UNIT * 10])
        # Building the obstacle 45
        self.obstacle45 = self.canvas.create_rectangle(
            obstacle45_center[0] - block_size, obstacle45_center[1] - block_size,  # Top left corner
            obstacle45_center[0] + block_size, obstacle45_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 45 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle45 = [self.canvas.coords(self.obstacle45)[0] + 3,
                                  self.canvas.coords(self.obstacle45)[1] + 3,
                                  self.canvas.coords(self.obstacle45)[2] - 3,
                                  self.canvas.coords(self.obstacle45)[3] - 3]

        # Obstacle 46
        # Defining the center of obstacle 46
        obstacle46_center = self.origin + np.array([UNIT * 19, UNIT * 10])
        # Building the obstacle 46
        self.obstacle46 = self.canvas.create_rectangle(
            obstacle46_center[0] - block_size, obstacle46_center[1] - block_size,  # Top left corner
            obstacle46_center[0] + block_size, obstacle46_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 46 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle46 = [self.canvas.coords(self.obstacle46)[0] + 3,
                                  self.canvas.coords(self.obstacle46)[1] + 3,
                                  self.canvas.coords(self.obstacle46)[2] - 3,
                                  self.canvas.coords(self.obstacle46)[3] - 3]

        # Obstacle 47
        # Defining the center of obstacle 47
        obstacle47_center = self.origin + np.array([UNIT * 19, UNIT * 9])
        # Building the obstacle 47
        self.obstacle47 = self.canvas.create_rectangle(
            obstacle47_center[0] - block_size, obstacle47_center[1] - block_size,  # Top left corner
            obstacle47_center[0] + block_size, obstacle47_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 47 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle47 = [self.canvas.coords(self.obstacle47)[0] + 3,
                                  self.canvas.coords(self.obstacle47)[1] + 3,
                                  self.canvas.coords(self.obstacle47)[2] - 3,
                                  self.canvas.coords(self.obstacle47)[3] - 3]

        # Obstacle 48
        # Defining the center of obstacle 48
        obstacle48_center = self.origin + np.array([UNIT * 19, UNIT * 8])
        # Building the obstacle 48
        self.obstacle48 = self.canvas.create_rectangle(
            obstacle48_center[0] - block_size, obstacle48_center[1] - block_size,  # Top left corner
            obstacle48_center[0] + block_size, obstacle48_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 48 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle48 = [self.canvas.coords(self.obstacle48)[0] + 3,
                                  self.canvas.coords(self.obstacle48)[1] + 3,
                                  self.canvas.coords(self.obstacle48)[2] - 3,
                                  self.canvas.coords(self.obstacle48)[3] - 3]

        # Obstacle 49
        # Defining the center of obstacle 49
        obstacle49_center = self.origin + np.array([UNIT * 11, UNIT * 23])
        # Building the obstacle 49
        self.obstacle49 = self.canvas.create_rectangle(
            obstacle49_center[0] - block_size, obstacle49_center[1] - block_size,  # Top left corner
            obstacle49_center[0] + block_size, obstacle49_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 49 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle49 = [self.canvas.coords(self.obstacle49)[0] + 3,
                                  self.canvas.coords(self.obstacle49)[1] + 3,
                                  self.canvas.coords(self.obstacle49)[2] - 3,
                                  self.canvas.coords(self.obstacle49)[3] - 3]

        # Obstacle 50
        # Defining the center of obstacle 50
        obstacle50_center = self.origin + np.array([UNIT * 10, UNIT * 23])
        # Building the obstacle 50
        self.obstacle50 = self.canvas.create_rectangle(
            obstacle50_center[0] - block_size, obstacle50_center[1] - block_size,  # Top left corner
            obstacle50_center[0] + block_size, obstacle50_center[1] + block_size,  # Bottom right corner
            outline='grey', fill='#000000')
        # Saving the coordinates of obstacle 50 according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_obstacle50 = [self.canvas.coords(self.obstacle50)[0] + 3,
                                  self.canvas.coords(self.obstacle50)[1] + 3,
                                  self.canvas.coords(self.obstacle50)[2] - 3,
                                  self.canvas.coords(self.obstacle50)[3] - 3]

        # create oval
        self.oval_center = self.origin + np.array([UNIT * end_loc[0], UNIT * end_loc[1]])
        self.oval = self.canvas.create_oval(
            self.oval_center[0] - block_size, self.oval_center[1] - block_size,
            self.oval_center[0] + block_size, self.oval_center[1] + block_size,
            fill='yellow')

        # create red rect
        self.agent_center = self.origin + np.array([UNIT * start_loc[0], UNIT * start_loc[1]])
        self.rect = self.canvas.create_rectangle(
            self.agent_center[0] - block_size, self.agent_center[1] - block_size,
            self.agent_center[0] + block_size, self.agent_center[1] + block_size,
            fill='red')

        # pack all
        self.canvas.pack()
        self.cors = []
        # for s in range(hell_size):
        #     self.cors.append(self.canvas.coords(self.hell[s]))

    def reset(self):  # 初始化智能体agent
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect)  # 画布清除
        self.origin = np.array([20, 20])
        # create_rectangle(x0, y0, x1, y1, options)
        self.canvas.create_rectangle(
            self.agent_center[0] - block_size, self.agent_center[1] - block_size,
            self.agent_center[0] + block_size, self.agent_center[1] + block_size,
            fill='blue')
        self.rect = self.canvas.create_rectangle(
            self.agent_center[0] - block_size, self.agent_center[1] - block_size,
            self.agent_center[0] + block_size, self.agent_center[1] + block_size,
            fill='red')
        # Clearing the dictionary and the i
        self.d = {}
        self.i = 0
        # return observation
        a = self.canvas.coords(self.rect)[:2]
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (
                    MAZE_H * UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        # print(base_action)
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        # Writing in the dictionary coordinates of found route
        self.d[self.i] = self.canvas.coords(self.rect)

        next_coords = self.d[self.i]  # next_coords

        # Updating key for the dictionary
        self.i += 1
        # reward function
        if next_coords == self.canvas.coords(self.oval):  # 到达终点
            reward = 1
            done = True
            # Filling the dictionary first time
            if self.c == True:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)

            # Checking if the currently found route is shorter
            if len(self.d) < len(self.f):
                # Saving the number of steps for the shortest route
                self.shortest = len(self.d)
                # Clearing the dictionary for the final route
                self.f = {}
                # Reassigning the dictionary
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]

            # Saving the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)
        elif next_coords in [self.obstacle1,
                             self.obstacle2,
                             self.obstacle3,
                             self.obstacle4,
                             self.obstacle5,
                             self.obstacle6,
                             self.obstacle7,
                             self.obstacle8,
                             self.obstacle9,
                             self.obstacle10,
                             self.obstacle11,
                             self.obstacle12,
                             self.obstacle13,
                             self.obstacle14,
                             self.obstacle15,
                             self.obstacle16,
                             self.obstacle17,
                             self.obstacle18,
                             self.obstacle19,
                             self.obstacle20,
                             self.obstacle21,
                             self.obstacle22,
                             self.obstacle23,
                             self.obstacle24,
                             self.obstacle25,
                             self.obstacle26,
                             self.obstacle27,
                             self.obstacle28,
                             self.obstacle29,
                             self.obstacle30,
                             self.obstacle31,
                             self.obstacle32,
                             self.obstacle33,
                             self.obstacle34,
                             self.obstacle35,
                             self.obstacle36,
                             self.obstacle37,
                             self.obstacle38,
                             self.obstacle39,
                             self.obstacle40,
                             self.obstacle41,
                             self.obstacle42,
                             self.obstacle43,
                             self.obstacle44,
                             self.obstacle45,
                             self.obstacle46,
                             self.obstacle47,
                             self.obstacle48,
                             self.obstacle49,
                             self.obstacle50]:  #
            reward = -1
            done = True
            # Clearing the dictionary and the i
            self.d = {}
            self.i = 0
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)

        return s_, reward, done

    def render(self, delay):
        time.sleep(delay)
        self.update()

    def final(self):
        # Deleting the agent at the end
        self.canvas.delete(self.rect)

        # Showing the number of steps
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)

        # Creating initial point
        self.initial_point = self.canvas.create_oval(
            self.rand[hell_size - 1][0] - 4, self.rand[hell_size - 1][0] - 4,
            self.rand[hell_size - 1][0] + 4, self.rand[hell_size - 1][0] + 4,
            fill='blue', outline='blue')

        # Filling the route
        for j in range(len(self.f)):
            # Showing the coordinates of the final route
            print(self.f[j])
            self.track = self.canvas.create_oval(
                self.f[j][0] - 3 + self.origin[0] / 2 - 4, self.f[j][1] - self.origin[0] / 2 - 4,
                self.f[j][0] - 3 + self.origin[0] / 2 + 4, self.f[j][1] - self.origin[0] / 2 + 4,
                fill='blue', outline='blue')
            # Writing the final route in the global variable a
            a[j] = self.f[j]


# Returning the final dictionary with route coordinates
# Then it will be used in agent_brain.py
def final_states():
    return a


if __name__ == '__main__':
    env = Maze_pravite()
    env.mainloop()
