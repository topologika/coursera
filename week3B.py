import numpy as np
import pandas

data = pandas.read_csv('data-logistic.csv', header = None)
y=np.array(data[0])
X=np.array(data[[1,2]])
