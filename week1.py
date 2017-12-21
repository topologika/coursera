import numpy as np
import pandas

# Week 1.1

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
# print(data[:10])
# print(data.head())
count=data['Sex'].value_counts(dropna=False)
a=data['Survived'].value_counts(dropna=False)
a.values[1]/sum(a.values)*100
age=data['Age'].value_counts(dropna=False)
data['Age'].mean()
data['Age'].median()
data.corr()
name=data[data.Sex=='female'].Name
data[data.Sex=='female'].Name.str.split("\. ",n=1).str[1].str.split(' \(').str.get(-1).str.replace('\(|\)','').str.split(n=1).str[0].value_counts()

# Week 1.2

from sklearn.tree import DecisionTreeClassifier
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)
importances = clf.feature_importances_

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
week2=data.loc[:,['Pclass','Fare','Age','Sex','Survived']].dropna(axis=0)
week2['NSex']=(week2['Sex']=='male')
        
x=np.array(week2[['Pclass','Fare','Age','NSex']])
y=np.array(week2['Survived'])

clf = DecisionTreeClassifier()
clf.random_state=241
clf.fit(x, y)

importances = clf.feature_importances_

