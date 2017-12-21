import numpy as np
import pandas
from sklearn import metrics # for scoring = "accuracy"
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

kf = KFold(n_splits=5,shuffle=True,random_state=42)

columns=['Class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

data = pandas.read_csv('wine.data', header = None, names = list(columns))
dataN = pandas.read_csv('wineN.data')

y=np.array(data['Class'])
X=np.array(data[['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']])
# 
# for k in range(1,51):
#     neigh = KNeighborsClassifier(n_neighbors=k)
#     for train_index, test_index in kf.split(X):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         print(cross_val_score(neigh, X, y, cv=kf))
#         #neigh.fit(X, y)
#
result=[0]
for k in range(1,51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(neigh, X, y, cv=kf, scoring='accuracy')
    result.append(score.mean())
#    print(score,score.mean())
npresult=np.array(result)
print(npresult.argmax(),npresult.max())
scoring='neg_mean_squared_error'
X = scale(X)

# здесь скалируем
result=[0]
for k in range(1,51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(neigh, X, y, cv=kf, scoring='accuracy')
    result.append(score.mean())
#    print(score,score.mean())
npresult=np.array(result)
print(npresult.argmax(),npresult.max())

# Задаение 2
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets # для задания 2
kf = KFold(n_splits=5,shuffle=True,random_state=42)

boston=datasets.load_boston()
X=scale(boston.data)
neigh = KNeighborsRegressor(n_neighbors=5,weights='distance',metric='minkowski')
score=cross_val_score(neigh, X, boston.target, cv=kf, scoring='neg_mean_squared_error')
result=[]
argument=[]
for p in np.linspace(1,10,200):
    neigh.set_params(p=p)
    score=cross_val_score(neigh, X, boston.target, cv=kf, scoring='neg_mean_squared_error')
    result.append(score.mean())
    argument.append(p)

npresult=np.array(result)
print(npresult.max(),argument[npresult.argmax()])