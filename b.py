import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt

my_dict = {
    0: [1.0, 23.0, 17.0, 37.0],
    1: [2.0, 43.0, 17.0, 57.0],
    2: [3.0, 23.0, 17.0, 37.0],
    3: [4.0, 43.0, 17.0, 57.0],
    4: [6.0, 63.0, 17.0, 77.0]
}

points = [(v[0], v[1]) for k, v in my_dict.items()]
np_array = np.array(points)
x = np.array([i[0] for i in points])
y = np.array([i[1] for i in points])
print(np_array)
print(x,y)
x, y = np.array(zip(*points))

spl = make_interp_spline(x, y, k=3)

xs = np.linspace(min(x), max(x), 200)
ys = spl(xs)

plt.plot(x, y, 'o') # 绘制原始点
plt.plot(xs, ys)    # 绘制平滑曲线
plt.show()
print(x,y)
