import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

import time


"""Heat map plots"""
# import seaborn as sns
# sns.set_theme()


# grad = []
# norm = []
# c = []
# for N in range (20,2000,10):
# 	tempg = []
# 	tempn = []
# 	for P in range(5,20,2):
# 		np.random.seed(P)

# 		X = pd.DataFrame(np.random.randint(0,100,size=(N, P)))
# 		y = 4*X + 7

# 		# print(X)

# 		LR = LinearRegression(fit_intercept=True)
# 		# c.append([N,P])
# 		# start_time = time.time()
# 		# LR.fit_vectorised(X, y, batch_size=N//10) 
# 		# end_time = time.time()

# 		# tempg.append(end_time-start_time)

# 		start_time = time.time()
# 		LR.fit_normal(X, y) 
# 		end_time = time.time()

# 		tempn.append(end_time-start_time)

# 	# grad.append(tempg)
# 	norm.append(tempn)

# corr = pd.DataFrame(norm)
# # corr = pd.DataFrame(grad)

# fig, ax = plt.subplots(figsize=(11, 9))

# sns.heatmap(corr)

# # yticks = [N for N in range (20,2000,10)]
# # xticks = [P for P in range(5,20,2)]
# # plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
# # plt.xticks(plt.xticks()[0], labels=xticks)
# plt.show()

""""""


"""Normal plt plots"""
# a = []
# b = []
# c = []
# for i in range (10,200):
# 	N = 15

# 	X = pd.DataFrame(np.random.randn(N, i))
# 	y = pd.Series(np.random.randn(N))

# 	LR = LinearRegression(fit_intercept=True)
# 	c.append(i)
# 	start_time = time.time()
# 	LR.fit_vectorised(X, y,batch_size=2) 
# 	end_time = time.time()

# 	a.append(end_time-start_time)

# 	start_time = time.time()
# 	LR.fit_normal(X, y) 
# 	end_time = time.time()

# 	b.append(end_time-start_time)

# plt.plot(c,a,label = 'Gradient Descent')
# plt.plot(c,b,label = 'Normal Equation')
# plt.legend(loc = 'best')
# plt.show()


a = []
b = []
c = []
for i in range (10,10000,10):

	X = pd.DataFrame(np.random.randn(i, 9))
	y = pd.Series(np.random.randn(i))

	LR = LinearRegression(fit_intercept=True)
	c.append(i)
	start_time = time.time()
	LR.fit_vectorised(X, y,batch_size=2) 
	end_time = time.time()

	a.append(end_time-start_time)

	start_time = time.time()
	LR.fit_normal(X, y) 
	end_time = time.time()

	b.append(end_time-start_time)

plt.plot(c,a,label = 'Gradient Descent')
plt.plot(c,b,label = 'Normal Equation')
plt.legend(loc = 'best')
plt.show()
""""""