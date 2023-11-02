import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate
from sklearn import svm, tree

iris = datasets.load_iris()

X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print(X.head())
y = pd.Series(iris['target_names'][iris['target']])
# y = pd.get_dummies(y)

linear_svm = svm.SVC(C=1, kernel='linear')
linear_scores = cross_validate(linear_svm, X, y, cv=5, scoring='accuracy')

print("线性核：")
print(linear_scores['test_score'].mean())

rbf_svm = svm.SVC(C=1)
rbf_scores = cross_validate(rbf_svm, X, y, cv=5, scoring='accuracy')

print("高斯核：")
print(rbf_scores['test_score'].mean())