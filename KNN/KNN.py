import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
u = pd.read_csv("C:\\python\\git\\KNN\\Classified Data")
#print(u.head())
scaler = StandardScaler()
scaler.fit(u.drop('TARGET CLASS',axis=1))
scaler_feature = scaler.transform(u.drop('TARGET CLASS',axis=1))
u_feature = pd.DataFrame(scaler_feature,columns=u.columns[:-1])
#print(u_feature)
#print(scaler_feature)
#print(u.columns)
x_train, x_test, y_train, y_test = train_test_split(scaler_feature,u['TARGET CLASS'])
#print(x_test)
KNN = KNeighborsClassifier(n_neighbors=1) #choose the k value such that it gives the best accuracy
KNN.fit(x_train,y_train)
predict = KNN.predict(x_test)
#print(predict)
#print(confusion_matrix(y_test,predict))
#print(classification_report(y_test,predict))
