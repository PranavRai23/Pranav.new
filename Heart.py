from cmath import log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix,plot_precision_recall_curve

df=pd.read_csv("C:/Users/DELL/heart.csv")
df['target'].unique()
df.isnull().sum()
correlt=df.corr()
#plt.figure(figsize=(12,8),dpi=100)
# sns.heatmap(correlt,annot=True)
X=df.drop('target',axis=1)
y=df['target']
# df['target'].value_counts()
# sns.set_theme(style="darkgrid")
# sns.countplot(x='target',data=df)
# pair=sns.pairplot(df[['age','trestbps', 'chol','thalach','target']],hue='target')
# plt.show()

# Train_Test_Split and scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
scaler= StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

# import model

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
log_model=LogisticRegressionCV()
log_model.fit(scaled_X_train, y_train)
# log_model.get_params()
# log_model.coef_[0]
# coefs=pd.Series(index=X.columns,data=log_model.coef_[0])
#print(coefs)

# coefs=coefs.sort_values()
# plt.figure(figsize=(8,4),dpi=200)
# sns.barplot(x=coefs,y=coefs.values)

y_pred=log_model.predict(scaled_X_test)
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(log_model, scaled_X_test, y_test)
print(classification_report(y_test,y_pred))
plt.show()
