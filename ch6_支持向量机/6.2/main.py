from sklearn import svm
from sklearn.model_selection import cross_val_score

X=[
    [1.   , 2.   , 1.   , 0.   , 2.   , 1.   , 0.697, 0.46 ],
    [2.   , 2.   , 0.   , 0.   , 2.   , 1.   , 0.774, 0.376],
    [2.   , 2.   , 1.   , 0.   , 2.   , 1.   , 0.634, 0.264],
    [1.   , 2.   , 0.   , 0.   , 2.   , 1.   , 0.608, 0.318],
    [0.   , 2.   , 1.   , 0.   , 2.   , 1.   , 0.556, 0.215],
    [1.   , 1.   , 1.   , 0.   , 1.   , 0.   , 0.403, 0.237],
    [2.   , 1.   , 1.   , 1.   , 1.   , 0.   , 0.481, 0.149],
    [2.   , 1.   , 1.   , 0.   , 1.   , 1.   , 0.437, 0.211],
    [2.   , 1.   , 0.   , 1.   , 1.   , 1.   , 0.666, 0.091],
    [1.   , 0.   , 2.   , 0.   , 0.   , 0.   , 0.243, 0.267],
    [0.   , 0.   , 2.   , 2.   , 0.   , 1.   , 0.245, 0.057],
    [0.   , 2.   , 1.   , 2.   , 0.   , 0.   , 0.343, 0.099],
    [1.   , 1.   , 1.   , 1.   , 2.   , 1.   , 0.639, 0.161],
    [0.   , 1.   , 0.   , 1.   , 2.   , 1.   , 0.657, 0.198],
    [2.   , 1.   , 1.   , 0.   , 1.   , 0.   , 0.36 , 0.37 ],
    [0.   , 2.   , 1.   , 2.   , 0.   , 1.   , 0.593, 0.042],
    [1.   , 2.   , 0.   , 1.   , 1.   , 1.   , 0.719, 0.103]
]
y=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]

print("-"*20+"线性核"+"-"*20)
clf1=svm.SVC(C=1,kernel='linear')
print("交叉验证评分",cross_val_score(clf1,X,y,cv=5,scoring='accuracy').mean())
clf1.fit(X,y)
print("支持向量数目",clf1.n_support_.sum())
print("支持向量",clf1.support_vectors_)

print("-"*20+"高斯核"+"-"*20)
clf2=svm.SVC(C=1,kernel='rbf')
print("交叉验证评分",cross_val_score(clf2,X,y,cv=5,scoring='accuracy').mean())
clf2.fit(X,y)
print("支持向量数目",clf2.n_support_.sum())
print("支持向量",clf2.support_vectors_)
