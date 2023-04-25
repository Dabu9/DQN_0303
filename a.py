import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

x = np.array([0, 2, 2, 3, 4])
y = np.array([0, 3, 5, 3, 1])

# 计算平滑曲线的坐标
tck, u = interpolate.splprep([x, y], s=0)
u_new = np.linspace(u.min(), u.max(), 1000)
x_new, y_new = interpolate.splev(u_new, tck)

# 绘制原始数据点和平滑曲线
plt.plot(x, y, 'ro', label='Original points')
plt.plot(x_new, y_new, 'b-', label='Smoothed curve')

# 绘制原始线段
for i in range(len(x)-1):
    plt.plot([x[i], x[i+1]], [y[i], y[i+1]], 'g--', label='Original line segments')

plt.legend()
plt.show()
