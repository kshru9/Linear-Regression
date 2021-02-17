import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

N = 30
P = 5
X = [[i*j for i in range(1,P+1)]  for j in range(1,N+1)]
y = [5]*N

X = pd.DataFrame(X)
y = pd.Series(y)

# print(X,y)

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X, y,batch_size=1, n_iter=500, lr=0.000001, lr_type='constant')
y_hat = LR.predict(X)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))