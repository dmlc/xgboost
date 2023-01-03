import numpy as np
import pandas
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
y = y.astype(np.int32)
df = pandas.DataFrame(data=X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
class_id_to_name = {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}
df['class'] = np.vectorize(class_id_to_name.get)(y)
df.to_csv('./iris.csv', float_format='%.1f', header=False, index=False)
