import numpy as np
import math
import matplotlib.pyplot as plt

data_x = [[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
          [0.481, 0.149], [0.437, 0.211],
          [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198],
          [0.360, 0.370], [0.593, 0.042], [0.719, 0.103]]
data_y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

mu_0 = np.mat([0., 0.]).T
mu_1 = np.mat([0., 0.]).T
count_0 = 0
count_1 = 0
for i in range(len(data_x)):
    x = np.mat(data_x[i]).T
    if data_y[i] == 1:
        mu_1 = mu_1 + x
        count_1 = count_1 + 1
    else:
        mu_0 = mu_0 + x
        count_0 = count_0 + 1
mu_0 = mu_0 / count_0
mu_1 = mu_1 / count_1

S_w = np.mat([[0, 0], [0, 0]])
for i in range(len(data_x)):
    # 注意：西瓜书的输入向量是列向量形式
    x = np.mat(data_x[i]).T
    if data_y[i] == 0:
        S_w = S_w + (x - mu_0) * (x - mu_0).T
    else:
        S_w = S_w + (x - mu_1) * (x - mu_1).T

u, sigmav, vt = np.linalg.svd(S_w)
sigma = np.zeros([len(sigmav), len(sigmav)])
for i in range(len(sigmav)):
    sigma[i][i] = sigmav[i]
sigma = np.mat(sigma)
S_w_inv = vt.T * sigma.I * u.T
w = S_w_inv * (mu_0 - mu_1)

w_0 = w[0, 0]
w_1 = w[1, 0]
tan = w_1 / w_0
sin = w_1 / math.sqrt(w_0 ** 2 + w_1 ** 2)
cos = w_0 / math.sqrt(w_0 ** 2 + w_1 ** 2)

print(w_0, w_1)

for i in range(len(data_x)):
    if data_y[i] == 0:
        plt.plot(data_x[i][0], data_x[i][1], "go")
    else:
        plt.plot(data_x[i][0], data_x[i][1], "b^")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Discriminant Analysis')
plt.plot(mu_0[0, 0], mu_0[1, 0], "ro")
plt.plot(mu_1[0, 0], mu_1[1, 0], "r^")
plt.plot([-0.1, 0.1], [-0.1 * tan, 0.1 * tan])

for i in range(len(data_x)):
    x = np.mat(data_x[i]).T
    ell = w.T * x
    ell = ell[0, 0]
    if data_y[i] == 0:
        plt.scatter(cos * ell, sin * ell, marker='o', edgecolors='g')
    else:
        plt.scatter(cos * ell, sin * ell, marker='^', edgecolors='b')
plt.show()
