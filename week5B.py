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
from sklearn.ensemble import RandomForestClassifier


#%matplotlib inline

data = pandas.read_csv('gbm-data.csv').values
y=data[:,0]
X=data[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=24)



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

a=np.array(train_loss_learning_rate)
b=np.array(test_loss_learning_rate)

farbe = ['orange','turquoise', 'blue','gray','magenta']
learning_rate = [1, 0.5, 0.3, 0.2, 0.1]

plt.figure(num=1)
for i in range(5):
    plt.plot(a[i],color=farbe[i],label=str(learning_rate[i]))
plt.legend(loc='upper left')    
plt.show()

plt.figure(num=2)
for i in range(5):
    plt.plot(b[i],color=farbe[i],label=str(learning_rate[i]))

plt.legend(loc='upper left')    
plt.show()

# Номера итераций с минимальным значением функций потерь на обучающей выборке
for i in range(5):
    print(a[i].min(), a[i].argmin()+1)
# на тестовой
for i in range(5):
    print(b[i].min(), b[i].argmin()+1)


clf = RandomForestClassifier(n_estimators=19, random_state=241)
clf.fit(X_train,y_train) 
pred = clf.predict_proba(X_test)
result = log_loss(y_test,pred)
print(result)

#       plt.plot(train_loss, 'r', linewidth=2)
#        plt.plot(train_loss, 'g', linewidth=2)
#        plt.legend(['test', 'train'])        
#        print(y)
        
#    print(GBC.staged_decision_function(X_train))
#    y_pred_train=GBC.staged_decision_function(X_train)
#    y_pred_test=GBC.staged_decision_function(X_test)


