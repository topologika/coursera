import numpy as np
from numpy import linalg as LA
import pandas
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

data = pandas.read_csv('classification.csv')
classification_true=np.array(data['true'])
classification_pred=np.array(data['pred'])

data = pandas.read_csv('scores.csv')
scores_true=np.array(data['true'])
scores_pred=np.array(data[['score_logreg','score_svm','score_knn','score_tree']])

TP=np.dot(classification_true,classification_pred)
FP=np.dot(1-classification_true,classification_pred)
FN=np.dot(classification_true,1-classification_pred)
TN=np.dot(1-classification_true,1-classification_pred)
print('{} {} {} {}'.format(TP,FP,FN,TN))

res_accuracy_score = accuracy_score(classification_true,classification_pred)
res_precision_score = precision_score(classification_true,classification_pred)
res_recall_score = recall_score(classification_true,classification_pred)
res_f1_score = f1_score(classification_true,classification_pred)

print('{} {} {} {}'.format(res_accuracy_score,res_precision_score,res_recall_score,res_f1_score))




