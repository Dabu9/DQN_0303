import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

my_dict = {
    0: [3.0, 23.0, 17.0, 37.0],
    1: [3.0, 43.0, 17.0, 57.0],
    2: [3.0, 23.0, 17.0, 37.0],
    3: [3.0, 43.0, 17.0, 57.0],
    4: [3.0, 63.0, 17.0, 77.0]
}
points = []
for key in my_dict:
    points.append((my_dict[key][0]+ np.random.normal(scale=1e-6), my_dict[key][1]))

print(points)

x, y = zip(*points)

print(x)
spl = make_interp_spline(x, y, k=3)

xs = np.linspace(min(x), max(x), 200)
ys = spl(xs)

plt.plot(x, y, 'o') # 绘制原始点
plt.plot(xs, ys)    # 绘制平滑曲线
plt.show()
