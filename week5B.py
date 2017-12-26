import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn import metrics # for scoring = "accuracy"
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
#%matplotlib inline

data = pandas.read_csv('gbm-data.csv').values
y=data[:,0]
X=data[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=24)

plt.figure()


train_loss_learning_rate=[]
test_loss_learning_rate=[]

for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    GBC=GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=250, verbose=True, random_state=241)
    GBC.fit(X_train,y_train)
    train_loss=[]
    test_loss=[]
    for i, y in enumerate(GBC.staged_decision_function(X_train)):
        y_ = 1 / (1 + np.exp(-y))
        train_loss.append(log_loss(y_train,y_))
    train_loss_learning_rate.append(train_loss)
    for i, y in enumerate(GBC.staged_decision_function(X_test)):
        y_ = 1 / (1 + np.exp(-y))
        test_loss.append(log_loss(y_test,y_))
    test_loss_learning_rate.append(test_loss)

#        plt.plot(train_loss, 'r', linewidth=2)
#        plt.plot(train_loss, 'g', linewidth=2)
#        plt.legend(['test', 'train'])        
#        print(y)
        
#    print(GBC.staged_decision_function(X_train))
#    y_pred_train=GBC.staged_decision_function(X_train)
#    y_pred_test=GBC.staged_decision_function(X_test)


