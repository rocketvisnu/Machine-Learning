import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
u = pd.read_csv("C:\\python\\git\\Linear Regression\\titanic_train.csv")
#print(u.head())
#print(u.columns)
x = u[['PassengerId','Pclass']]
y = u[['Fare']]
#print(x)
x_train,x_test,y_train,y_test = train_test_split(x,y)
#print(x_train)
lr = LinearRegression()
lr.fit(x_train,y_train)
predict = lr.predict(x_test)
print(lr.intercept_)
plt.scatter(y_test,predict)