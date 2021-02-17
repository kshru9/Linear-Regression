
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

m = 100
X = [[i] for i in range(m)]

y = [5*i+2 for i in range(m)]

X = pd.DataFrame(X)
y = pd.Series(y)

for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    # LR.fit_vectorised(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    # LR.fit_non_vectorised(X, y, lr=0.0001)
    # LR.fit_normal(X,y)
    print(LR.fit_vectorised(X,y,batch_size=2, lr=0.000001, n_iter=1000))
    
    
    y_hat = LR.predict(X)

    # print('RMSE: ', rmse(y_hat, y))
    # print('MAE: ', mae(y_hat, y))