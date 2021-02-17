import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

# x = np.array([i*np.pi/180 for i in range(60,300,4)])
# np.random.seed(10)  #Setting seed for reproducibility
# y = 4*x + 7 + np.random.normal(0,3,len(x))

def generate_data(N):
    x = np.array([i*np.pi/180 for i in range(60, (60 + (N*4))+1 ,4)]) 
    np.random.seed(10) #Setting seed for reproducibility 
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    return x,y

def vary_degree(list_deg,x,y,fit_method="vec"):
    l = []
    for degree in list_deg:
        include_bias = True
        poly = PolynomialFeatures(degree,include_bias = include_bias)
        X_trans= []
        for i in range(len(x)):
            ar = np.array([x[i]])
            X_trans.append(poly.transform(ar))
        
        # print(len(X_trans),len(y))
        # print(pd.DataFrame(X_trans))
        X = X_trans
        LR = LinearRegression(fit_intercept=True)

        if (fit_method=="normal"):
            thetas = LR.fit_normal(pd.DataFrame(X),pd.Series(y))
        elif (fit_method=="non_vec"):
            thetas = LR.fit_non_vectorised(pd.DataFrame(X),pd.Series(y),batch_size=1)
        elif (fit_method=="vec"):
            thetas = LR.fit_vectorised(pd.DataFrame(X),pd.Series(y),batch_size=1)
        else:
            thetas = LR.fit_autograd(pd.DataFrame(X),pd.Series(y),batch_size=1)
        
        # print(thetas)
        l.append(np.linalg.norm(np.array(thetas)))

    return l

def plot_graph(a,b):
    plt.plot(a,b)
    plt.xlabel("Degree of fitted polynmial")
    plt.ylabel("Magnitude of theta")
    plt.yscale("log")
    plt.savefig("./figures/q5_plot.png", dpi=400)

x,y = generate_data(60)

# print(len(x),len(y))

a = [1,2,3,4,5,6,7,8,9]
b = vary_degree(a,x,y)

plot_graph(a,b)