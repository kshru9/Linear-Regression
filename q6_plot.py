import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

def generate_data(N):
    """generate data for the given number of samples"""
    x = np.array([i*np.pi/180 for i in range(60, (60 + (N*4))+1 ,4)]) 
    np.random.seed(10) #Setting seed for reproducibility 
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    return x,y

def vary_degree(list_deg,x,y,fit_method="vec"):
    """a fucntion that will calculate """
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

list_of_degrees = [1,3,5,7,9]

fig = plt.figure(figsize = (8,8))

fig_count = 1
for N in range(1,8,2):
    x,y = generate_data(N)
    theta = vary_degree(list_of_degrees, x,y)
    # plt.subplot(2,2,fig_count)
    plt.plot(list_of_degrees,theta, label="N= "+str(N))

plt.xlabel("degrees")
plt.ylabel("thetas")

plt.yscale("log")

plt.legend()
    # fig_count+=1
plt.savefig("./figures/q6_comb"+str(N)+".png", dpi=400)
    