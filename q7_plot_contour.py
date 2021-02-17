import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

N = 15
P = 1
X = np.random.rand(N, P)
y = 4*X+7
X = pd.DataFrame(X)
y = y.reshape(len(y),)
y = pd.Series(y)

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X,y,batch_size = 3,lr = 0.005,n_iter = 500)
t_0 = 1
t_1 = 2
LR.plot_line_fit(X,y,t_0,t_1)
# LR.plot_surface(X,y,1,2)