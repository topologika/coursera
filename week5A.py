import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn import metrics # for scoring = "accuracy"
from sklearn.model_selection import cross_val_score




kf = KFold(n_splits=5,shuffle=True,random_state=1)


# X = np.array([[1, 2], [3, 4], [5, 6]])
# y = np.array([-3, 1, 10])
# clf = RandomForestRegressor(n_estimators=100)
# clf.fit(X, y)
# predictions = clf.predict(X)

# print(r2_score([10, 11, 12], [9, 11, 12.1]))

data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data[['Sex','Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']]

y = data['Rings']

for trees in range(1,51):
    regr = RandomForestRegressor(n_estimators=trees, random_state=1)
    score=cross_val_score(regr, X, y, cv=kf, scoring='r2')
    print(score.mean(), trees)

#Sex,Length,Diameter,Height,WholeWeight,ShuckedWeight,VisceraWeight,ShellWeight,Rings