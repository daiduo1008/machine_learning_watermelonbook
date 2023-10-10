import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.linear_model import LogisticRegression

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 创建逻辑回归模型
logreg = LogisticRegression(max_iter=10000)

# 使用10折交叉验证估计错误率
scores_10_fold = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
mean_error_rate_10_fold = 1 - np.mean(scores_10_fold)

print(f'错误率（10折交叉验证）：{mean_error_rate_10_fold:.2f}')

# 使用留一法估计错误率
loo = LeaveOneOut()
scores_loo = cross_val_score(logreg, X, y, cv=loo, scoring='accuracy')
mean_error_rate_loo = 1 - np.mean(scores_loo)

print(f'错误率（留一法）：{mean_error_rate_loo:.2f}')
