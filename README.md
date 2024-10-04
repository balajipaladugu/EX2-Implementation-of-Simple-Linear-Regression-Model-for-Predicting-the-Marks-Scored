# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```

Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: PALADUGU VENKATA BALAJI
RegisterNumber:2305001024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
x_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_train,lr.predict(x_train),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
X_test
Y_test
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test, pred)
print(f'Mean Squared Error (MSE): {mse}')
```

## Output:
![image](https://github.com/user-attachments/assets/15e8a573-de42-4338-b318-8b7442d40ba4)
![image](https://github.com/user-attachments/assets/78d86d98-1403-4714-bec0-3161c29efb1b)
![image](https://github.com/user-attachments/assets/45b856ed-0498-470f-8168-0f6b129f432e)
![image](https://github.com/user-attachments/assets/20c7d379-8de5-43a9-84a3-afe6b3d96734)
![image](https://github.com/user-attachments/assets/666cff5e-6853-4080-b117-987adfc24ad2)
![image](https://github.com/user-attachments/assets/1cfe89bf-0040-4964-941e-56d66a306e41)
![image](https://github.com/user-attachments/assets/3f5e6ee5-93b6-4148-a1c9-5d761b9feec5)
![image](https://github.com/user-attachments/assets/f9f5235d-46d2-4b45-9192-68429badd0a2)



## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
