from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn import datasets
iris = datasets.load_iris()

#print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])

X = iris['data'][:, 3:]
y=(iris['target']==2).astype(np.int)


# Train a logisticregression classifier

clf=LogisticRegression()
clf.fit(X,y)
example = clf.predict(([[1.6]]))
print(example)

# Using matplotlib to plot the visualization

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)  
plt.plot(X_new, y_prob[:, 1], "g-", label="virginica")
plt.show()