# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.
2.Load the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Dharshini S
RegisterNumber: 212223110010 
*/
```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Placement_Data.csv')
dataset
```
![Screenshot 2024-10-13 181320](https://github.com/user-attachments/assets/a6bf148e-59e0-4a4e-8a02-5d5b71e4083e)
```
dataset = dataset.drop('sl_no',axis=1)
```
![Screenshot 2024-10-13 181330](https://github.com/user-attachments/assets/a67ff142-b9fe-4276-b146-9e427db109b8)
```
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
```
```
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
X
Y
```
![Screenshot 2024-10-13 181335](https://github.com/user-attachments/assets/cb9cc7d8-9596-4e21-b0a3-711c004d0b00)
```
theta = np.random.randn(X.shape[1])
y = Y
```
```
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
```
![Screenshot 2024-10-13 182025](https://github.com/user-attachments/assets/5ef49571-4835-4e22-9b88-34a1102738d4)
```
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
![Screenshot 2024-10-13 181404](https://github.com/user-attachments/assets/794a8573-418c-4965-bd8a-945d8f68ad5f)
```
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)

```
## Output:
![Screenshot 2024-10-13 182207](https://github.com/user-attachments/assets/9e2cd42c-b640-440a-9a36-ba882ffd2f02)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

