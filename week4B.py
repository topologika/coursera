import numpy as np
import pandas
from sklearn.decomposition import PCA

columns = ['AXP','BA','CAT','CSCO','CVX','DD','DIS','GE','GS','HD','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','PFE','PG','T','TRV','UNH','UTX','V','VZ','WMT','XOM']

close_prices = pandas.read_csv('close_prices.csv')
djia_index = pandas.read_csv('djia_index.csv')

pca = PCA(n_components=10)
pca.fit(close_prices[columns])

# Ответ на первый вопрос 4
pca.explained_variance_ratio_[:4].sum() #=0.92774295378364058

X_transformed = pca.transform(close_prices[columns])
pca1 = X_transformed[:,0]

# Ответ на второй вопрос 0.91
corrcoef = np.corrcoef([pca1,djia_index['^DJI']])

# Ответ на третий вопрос: V
print(columns[abs(pca.components_[0]).argmax()])


