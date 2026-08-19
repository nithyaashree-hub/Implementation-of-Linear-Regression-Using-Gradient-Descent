# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)
y = np.array([50, 45, 70, 80, 95], dtype=float)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
X = np.c_[np.ones(X.shape[0]), X]  # shape becomes (n_samples, n_features + 1)
n_features = X.shape[1]
weights = np.zeros(n_features)
learning_rate = 0.01
epochs = 1000
for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi

print("Trained Weights (including intercept):", weights)
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)

/*
Program to implement the linear regression using gradient descent.
Developed by: NITHYAA SHREE T
RegisterNumber:  212225220069
*/
```

## Output:
<img width="1920" height="1080" alt="Screenshot (458)" src="https://github.com/user-attachments/assets/18614d70-60db-4222-a4b2-d5859e3d542f" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
