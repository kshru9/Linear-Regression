import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures
import sys


X = np.array([1,2])
degree = int(sys.argv[1])
poly = PolynomialFeatures(degree,include_bias=True)
print(poly.transform(X))