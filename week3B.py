import numpy as np
from numpy import linalg as LA
import pandas
from sklearn.metrics import roc_auc_score


def a(x):
     return(1 / (1 + np.exp(-x.dot(w)))) 

data = pandas.read_csv('data-logistic.csv', header = None)
y=np.array(data[0])
X=np.array(data[[1,2]])
w=np.array([0,0])
k=0.1
C=0
epsilon=10**-5
maxIter = 10000
Y=y*(1-1/(1+np.exp(-y*X.dot(w))))
Z=X*np.array([Y,Y]).T
w1=w+k*Z.mean(0)-k*C*w
iter = 0
while iter < maxIter and LA.norm(w-w1) > epsilon:
    w=w1
    Y=y*(1-1/(1+np.exp(-y*X.dot(w))))
    Z=X*np.array([Y,Y]).T
    w1=w+k*Z.mean(0)-k*C*w
    iter+=1
w=w1
 
print(roc_auc_score(y, a(X)))  
