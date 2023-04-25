# Importing libraries
import numpy as np  # To deal with data in form of matrices
import tkinter as tk  # To build GUI
import time  # Time is needed to slow down the agent and to see how he runs
from PIL import Image, ImageTk  # For adding images into the canvas widget


# Global variable for dictionary with coordinates for the final route
a = {}
# 定义目标点坐标 将数组的值中设为8
m1 = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
      [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
      [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 8],
      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]

m2 = [[0, 1, 0, 0, 0],
      [0, 1, 0, 1, 8],
      [0, 1, 0, 1, 0],
      [0, 1, 0, 1, 0],
      [0, 0, 0, 1, 0]]

m3 = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 8, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
map_data = m1


# Setting the sizes for the environment
pixels = 20  # pixels
env_height = len(map_data)  # grid height
env_width = env_height  # grid width

# Creating class for the environment
class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.attributes("-topmost", 1)
        self.action_space = ['up', 'down', 'left', 'right', 'right-up', 'right-down', 'left-up', 'left-down']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('RL Q-learning')
        self.geometry('{0}x{1}'.format(env_height * pixels, env_height * pixels))
        self.build_environment()
        self.flag_times = 0
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

    # Function to build the environment
    def build_environment(self):
        self.canvas_widget = tk.Canvas(self, bg='white',
                                       height=env_height * pixels,
                                       width=env_width * pixels)

        # Uploading an image1 for background
        # img_background = Image.open("images/bg.png")
        # self.background = ImageTk.PhotoImage(img_background)
        # Creating background on the widget
        # self.bg = self.canvas_widget.create_image(0, 0, anchor='nw', image1=self.background)

        # 创建网格线
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')

        # 创建障碍物
        self.obstacles = []
        # An array to help with building rectangles
        self.o = np.array([pixels / 2, pixels / 2])
        # 遍历二维数组并创建障碍物
        for i, row in enumerate(map_data):
            for j, value in enumerate(row):
                if value == 1:
                    # 计算障碍物中心坐标
                    obstacle_center = np.array([j * pixels, i * pixels]) + self.o
                    obstacle_name = f"obstacle_{i}_{j}"
                    obstacle = self.canvas_widget.create_rectangle(obstacle_center[0] - 10,
                                                                   obstacle_center[1] - 10,
                                                                   obstacle_center[0] + 10,
                                                                   obstacle_center[1] + 10,
                                                                   outline='grey', fill='#000000',
                                                                   tags=(obstacle_name,)
                                                                   )
                    coords_obstacle = [self.canvas_widget.coords(obstacle)[0] + 3,
                                       self.canvas_widget.coords(obstacle)[1] + 3,
                                       self.canvas_widget.coords(obstacle)[2] - 3,
                                       self.canvas_widget.coords(obstacle)[3] - 3]
                    self.obstacles.append(coords_obstacle)
                elif value == 8:
                    flag_position = [j, i]
        self.obstacles_index = np.zeros(len(self.obstacles))
        # Creating an agent of Mobile Robot - red point
        self.agent = self.canvas_widget.create_oval(
            self.o[0] - 7, self.o[1] - 7,
            self.o[0] + 7, self.o[1] + 7,
            outline='#FF1493', fill='#FF1493')

        # Final Point - yellow point
        flag_center = self.o + np.array([pixels * flag_position[0], pixels * flag_position[1]])
        # Building the flag
        self.flag = self.canvas_widget.create_rectangle(
            flag_center[0] - 10, flag_center[1] - 10,  # Top left corner
            flag_center[0] + 10, flag_center[1] + 10,  # Bottom right corner
            outline='grey', fill='yellow')
        # Saving the coordinates of the final point according to the size of agent
        # In order to fit the coordinates of the agent
        self.coords_flag = [self.canvas_widget.coords(self.flag)[0] + 3,
                            self.canvas_widget.coords(self.flag)[1] + 3,
                            self.canvas_widget.coords(self.flag)[2] - 3,
                            self.canvas_widget.coords(self.flag)[3] - 3]

        # Packing everything
        self.canvas_widget.pack()

    # Function to reset the environment and start new Episode
    def reset(self):
        self.update()
        # time.sleep(0.5)

        # Updating agent
        self.canvas_widget.delete(self.agent)
        self.agent = self.canvas_widget.create_oval(
            self.o[0] - 7, self.o[1] - 7,
            self.o[0] + 7, self.o[1] + 7,
            outline='red', fill='red')

        # Clearing the dictionary and the i
        self.d = {}
        self.i = 0

        # Return observation
        state = (np.array(self.canvas_widget.coords(self.agent)[:2]) - np.array(self.canvas_widget.coords
                                                                                (self.flag)[:2])) / (
                            env_height * pixels)
        return state

    # Function to get the next observation and reward by doing next step
    def step(self, action):
        # Current state of the agent
        state = self.canvas_widget.coords(self.agent)
        base_action = np.array([0, 0])
        # 撞墙的标志,如果撞墙了定义为1
        bump_tag = 0

        # Updating next state according to the action
        # Action 'up'
        if action == 0:
            if state[1] >= pixels:
                base_action[1] -= pixels
            else:
                bump_tag = 1
        # Action 'down'
        elif action == 1:
            if state[1] < (env_height - 1) * pixels:
                base_action[1] += pixels
            else:
                bump_tag = 1
        # Action right
        elif action == 2:
            if state[0] < (env_width - 1) * pixels:
                base_action[0] += pixels
            else:
                bump_tag = 1
        # Action left
        elif action == 3:
            if state[0] >= pixels:
                base_action[0] -= pixels
            else:
                bump_tag = 1
        # Action right-up
        elif action == 4:
            if state[0] < (env_width - 1) * pixels and state[1] >= pixels:
                base_action[0] += pixels
                base_action[1] -= pixels
            else:
                bump_tag = 1
        # Action right-down
        elif action == 5:
            if state[0] < (env_width - 1) * pixels and state[1] < (env_height - 1) * pixels:
                base_action[0] += pixels
                base_action[1] += pixels
            else:
                bump_tag = 1
        # Action left-up
        elif action == 6:
            if state[0] >= pixels and state[1] >= pixels:
                base_action[0] -= pixels
                base_action[1] -= pixels
            else:
                bump_tag = 1
        # Action left-down
        elif action == 7:
            if state[0] >= pixels and state[1] < (env_height - 1) * pixels:
                base_action[0] -= pixels
                base_action[1] += pixels
            else:
                bump_tag = 1

        # Moving the agent according to the action
        self.canvas_widget.move(self.agent, base_action[0], base_action[1])

        # Writing in the dictionary coordinates of found route
        self.d[self.i] = self.canvas_widget.coords(self.agent)

        # Updating next state
        next_state = self.d[self.i]

        # Updating key for the dictionary
        self.i += 1

        agent_pos = np.array(self.canvas_widget.coords(self.agent)[:2])
        flag_pos = np.array(self.canvas_widget.coords(self.flag)[:2])
        distance_to_goal = np.linalg.norm(agent_pos - flag_pos) / (env_height * pixels)

        # Calculating the reward for the agent
        if bump_tag:
            reward = -0.5
            done = False
        elif next_state == self.coords_flag:
            time.sleep(0.1)
            print('--------------')
            reward = 1
            done = True
            self.flag_times += 1
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

        elif next_state in self.obstacles:
            index = self.obstacles.index(next_state)
            self.obstacles_index[index] += 1
            reward = -1
            done = True
            # next_state = 'obstacle'

            # Clearing the dictionary and the i
            self.d = {}
            self.i = 0

        else:
            reward = -0.01 * distance_to_goal
            done = False
        state = (np.array(self.canvas_widget.coords(self.agent)[:2]) - np.array(self.canvas_widget.coords
                                                                                (self.flag)[:2])) / (
                            env_height * pixels)
        return state, reward, done, self.flag_times

    # Function to refresh the environment
    def render(self):
        self.update()

    # Function to show the found route
    def final(self):
        # Deleting the agent at the end
        self.canvas_widget.delete(self.agent)

        # Showing the number of steps
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)

        # Creating initial point
        self.initial_point = self.canvas_widget.create_oval(
            self.o[0] - 4, self.o[1] - 4,
            self.o[0] + 4, self.o[1] + 4,
            fill='blue', outline='blue')

        # Filling the route
        for j in range(len(self.f) - 1):
            # Showing the coordinates of the final route
            print(self.f[j])
            # Drawing a line from the current point to the next point
            self.track = self.canvas_widget.create_line(
                self.f[j][0] + self.o[0], self.f[j][1] + self.o[1],
                self.f[j + 1][0] + self.o[0], self.f[j + 1][1] + self.o[1],
                fill='blue', width=3)
            # Writing the final route in the global variable a
            a[j] = self.f[j]


# Returning the final dictionary with route coordinates
# Then it will be used in agent_brain.py
def final_states():
    return a


# This we need to debug the environment
# If we want to run and see the environment without running full algorithm
if __name__ == '__main__':
    env = Environment()
    env.mainloop()
