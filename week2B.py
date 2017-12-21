# Здесь вторая часть недели 2
import numpy as np
from sklearn.linear_model import Perceptron
import pandas
from sklearn.metrics import accuracy_score

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = Perceptron(random_state=241)
clf.fit(X, y)
predictions = clf.predict(X)

data_train=pandas.read_csv('perceptron-train.csv', header = None)
X_train = np.array(data_train[[1,2]])
y_train = np.array(data_train[0])

data_test=pandas.read_csv('perceptron-test.csv', header = None)
X_test = np.array(data_test[[1,2]])
y_test = np.array(data_test[0])

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
r1=accuracy_score(y_test,predictions)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, y_train)

predictions = clf.predict(X_test_scaled)
r2=accuracy_score(y_test,predictions)

print(r2-r1)  # ответ к шестому, последнему по порядку, заданию недели 2