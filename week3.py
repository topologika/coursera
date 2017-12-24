#%% Задание 1

import numpy as np
import pandas
from sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv', header = None)
y=np.array(data[0])
X=np.array(data[[1,2]])

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, y)
print(clf.support_)

#%% Задание 2
import numpy as np
import pandas
from sklearn.svm import SVC
from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
             
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=241)

from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV 

grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups.data)

feature_mapping = vectorizer.get_feature_names()
#print(feature_mapping[700])
gs.fit(vectors, newsgroups.target)

for a in gs.grid_scores_:
    print(a.mean_validation_score) #— оценка качества по кросс-валидации
    print(a.parameters) # — значения параметров

clf = SVC(C=1, kernel='linear', random_state=241)
clf.fit(vectors, newsgroups.target)

a=clf.coef_.indices
b=abs(clf.coef_.data)
c=b.copy()
ind=c.argsort()
res=[]
for i in range(1,11):
    # print(i,ind[-i])
    # print(a[ind[-i]])
    print(feature_mapping[a[ind[-i]]])
    res.append(feature_mapping[a[ind[-i]]])
res.sort()
print(res)
for i in res:
    print(i)

#%%