import numpy as np
import pandas
from sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv', header = None)
y=np.array(data[0])
X=np.array(data[[1,2]])

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, y)
print(clf.support_)

# Задание 2
from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
             
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=241)

from sklearn.model_selection import GridSearchCV

grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

#gs.fit(X, y)