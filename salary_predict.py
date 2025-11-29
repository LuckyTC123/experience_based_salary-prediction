import pandas as pd
#first reading the file
data=pd.read_csv("sal.csv")
import joblib

#summarizing the variable in which data is stored
##print(data.head())
#print(data.info())
#print((data.describe()))
#print(data.shape)
#print(data.columns)
#print(data['Salary'])
#created the feature and target bins two containers

y=data['Salary']
x=data[['Experience Years']]

from sklearn.model_selection import train_test_split
##follwing function breaking x into two part and y into two parts
x_train,x_test,y_train,y_test=train_test_split(x,y)
##print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
from sklearn.linear_model import LinearRegression
#model=LinearRegression()

#training the most simplest  step just apply model.fit irrespective of any model
#model.fit(x_train,y_train)
#joblib.dump(model,"model_final.pkl")
model=joblib.load("model_final.pkl")
import numpy as np
print(model.coef_)
print(model.intercept_)
n=np.array([[2.33]])
print(model.predict(n))


