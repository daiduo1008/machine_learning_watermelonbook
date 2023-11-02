import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')


data= pd.DataFrame(np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
     [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
     [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
     [0.593,0.042],[0.719,0.103]]))


#X要是df类型，而y是se类型
X = data.iloc[:, [0]].values
y = data.iloc[:, 1].values

gamma = 10
C = 1

ax = plt.subplot()
set_ax_gray(ax)
ax.scatter(X, y, color='b', label='data')

for gamma in [1, 10, 100, 1000]:
    svr = svm.SVR(kernel='rbf', gamma=gamma, C=C)
    svr.fit(X, y)

    ax.plot(np.linspace(0.2, 0.8), svr.predict(np.linspace(0.2, 0.8).reshape(-1, 1)),
            label='gamma={}, C={}'.format(gamma, C))
ax.legend(loc='upper left')
ax.set_xlabel('密度')
ax.set_ylabel('含糖率')

plt.show()