import numpy as np
import pandas
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


enc = DictVectorizer()




salary_train = pandas.read_csv('salary-train.csv')
salary_test_mini = pandas.read_csv('salary-test-mini.csv')

salary_train['FullDescription']=salary_train['FullDescription'].str.lower()
salary_test_mini['FullDescription']=salary_test_mini['FullDescription'].str.lower()

salary_train['FullDescription']=salary_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
salary_test_mini['FullDescription']=salary_test_mini['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

salary_train['LocationNormalized'].fillna('nan', inplace=True)
salary_train['ContractTime'].fillna('nan', inplace=True)

salary_test_mini['LocationNormalized'].fillna('nan', inplace=True)
salary_test_mini['ContractTime'].fillna('nan', inplace=True)

vectorizer = TfidfVectorizer(min_df=5)
vectors = vectorizer.fit_transform(salary_train['FullDescription'])


X_train_categ = enc.fit_transform(salary_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(salary_test_mini[['LocationNormalized', 'ContractTime']].to_dict('records'))




#FullDescription,LocationNormalized,ContractTime,SalaryNormalized