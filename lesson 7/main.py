import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split




data = pd.read_csv("Data.csv")


x = data.iloc[:,:-1].values
print(x)

y = data.iloc[:,-1].values
print(y)



#take care of missing/nan data

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

imputer.fit(x[:,1:3])

x[:,1:3]=imputer.transform(x[:,1:3])
print(x)




#encoding categorical

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')

x=np.array(ct.fit_transform(x))
print(x)




#encoding using label encoders

lb=LabelEncoder()

y=lb.fit_transform(y)

print(y)




#normalizing

sc=StandardScaler()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)  #test size % to test,    to ensure no shuffeling -> random_state = 42 or 1 


print("x train")
print(x_train)
print("y train")
print(y_train)

print("x test")
print(x_test)
print("y test")
print(y_test)

x_train[:,:3]=sc.fit_transform(x_train[:,:3])

print(x_train[:,:3])