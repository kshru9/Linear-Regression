import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad 

from numpy.linalg import inv
# import plotly.graph_objects as go

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept

        self.thetas = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.theta_history = []
        self.cost_func_history = []

        self.x = None

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        if (self.fit_intercept == True):
            self.num_of_thetas = len(list(X.columns))+1
            self.thetas = pd.Series(np.zeros(self.num_of_thetas))
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        else:
            self.num_of_thetas = len(list(X.columns))
            self.thetas = pd.Series(np.random.randn(self.num_of_thetas))

        # print("self.num_of_thetas",self.num_of_thetas)
        # print(self.thetas)
        

        self.num_of_samples = len(X)
        self.num_of_iterations = n_iter
        self.learning_rate = lr
        self.lr_decay = lr_type

        self.theta_history.append(list(self.thetas))

        y =y.rename('y')
        dataset = pd.concat([X,y],axis=1)
        
        for i in range(self.num_of_iterations):
            if (self.lr_decay == "inverse"):
                self.learning_rate = lr / (i+1)

            for theta in range(self.num_of_thetas):

                epsilon = 0
                X_i,y_i = self.form_mini_batch(dataset,batch_size)
                m = batch_size
                for curr_sample in range(batch_size):
                    
                    if (self.fit_intercept):
                        temp = self.thetas[0]
                        for k in range(1,self.num_of_thetas):
                        # print("k:",k)
                            temp += X_i.iloc[curr_sample,k] * self.thetas[k]
                    else:
                        temp = 0
                        for k in range(self.num_of_thetas):
                        # print("k:",k)
                            temp += X_i.iloc[curr_sample,k] * self.thetas[k]
                    
                    epsilon += y_i[curr_sample] - (temp * (- X_i.iloc[curr_sample,theta]))
                
                self.thetas[theta] = self.thetas[theta] - ((self.learning_rate * 2 * epsilon * (1/m)))

                self.theta_history.append(list(self.thetas))

        return self.thetas

    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        if (self.fit_intercept == True):
            self.num_of_thetas = len(list(X.columns))+1
            self.thetas = pd.Series(np.random.randn(self.num_of_thetas))
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        else:
            self.num_of_thetas = len(list(X.columns))
            self.thetas = pd.Series(np.random.randn(self.num_of_thetas))

        self.num_of_samples = len(X)
        self.num_of_iterations = n_iter
        self.learning_rate = lr
        self.lr_decay = lr_type

        self.theta_history.append(list(self.thetas))

        y =y.rename('y')
        dataset = pd.concat([X,y],axis=1)

        for i in range(self.num_of_iterations):
            if (self.lr_decay == "inverse"):
                self.learning_rate = lr / (i+1)

            X_i,y_i = self.form_mini_batch(dataset,batch_size)
            m = batch_size

            temp = pd.Series(np.dot(X_i,self.thetas)) - y_i
            epsilon = pd.Series(np.dot(X_i.transpose(), temp))

            self.cost_func_history.append(np.linalg.norm(np.dot(X_i.transpose(), temp)))

            self.thetas = self.thetas - self.learning_rate * epsilon * (1/m)
            self.theta_history.append(list(self.thetas))

        return self.thetas

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        def cost_funct(thetas):
            epsilon = np.dot(X_il,thetas)-y_il
            return np.sum((epsilon)**2)/m

        
        if (self.fit_intercept == True):
            self.num_of_thetas = len(list(X.columns))+1
            self.thetas = pd.Series(np.random.randn(self.num_of_thetas))
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        else:
            self.num_of_thetas = len(list(X.columns))
            self.thetas = pd.Series(np.random.randn(self.num_of_thetas))

        self.num_of_samples = len(X)
        self.num_of_iterations = n_iter
        self.learning_rate = lr
        self.lr_decay = lr_type

        self.theta_history.append(list(self.thetas))

        y =y.rename('y')
        dataset = pd.concat([X,y],axis=1)
        
        gradient = grad(cost_funct)

        for iteration in range(self.num_of_iterations):
            if lr_type=='inverse':
                self.learning_rate = lr/(iteration+1)
            
            X_i,y_i = self.form_mini_batch(dataset,batch_size)
            X_il,y_il = X_i.to_numpy(),y_i.to_numpy()
            m = batch_size
            theta_il = self.thetas.to_numpy()

            # print(X_il,y_il,theta_il)
            temp_grad = gradient(theta_il)

            self.cost_func_history.append(temp_grad)
            
            self.thetas = self.thetas - self.learning_rate*temp_grad
            self.theta_history.append(list(self.thetas))

        return self.thetas

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''

        if (self.fit_intercept == True):
            self.num_of_thetas = len(list(X.columns))+1
            self.thetas = pd.Series(np.random.randn(self.num_of_thetas))
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        else:
            self.num_of_thetas = len(list(X.columns))
            self.thetas = pd.Series(np.random.randn(self.num_of_thetas))

        # temp = inv(X.transpose().dot(X))
        # temp1 = temp.dot(X.transpose())
        # self.thetas = temp1.dot(y)

        X_transpose = np.transpose(X)
        X_tr_dot_x = X_transpose.dot(X)
        temp1 = np.linalg.inv(X_tr_dot_x)
        temp2 = X_transpose.dot(y)
        self.thetas = temp1.dot(temp2)
        
        return self.thetas

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))

        if self.fit_intercept:
            X = pd.concat([bias,X],axis=1)

        return pd.Series(np.dot(X,self.thetas))

    def form_mini_batch(self, dataset,batch_size):
        sample = dataset.sample(n=batch_size)
        sample.reset_index(drop=True,inplace = True)
        y_s = sample.pop('y')
        X_s = sample
        return X_s,y_s

    def costfunction(self,X,y,theta):
        
        if self.fit_intercept:
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        
        # print(X,y,theta)
        
        temp = pd.Series(np.dot(X,theta)) - y
        # epsilon = pd.Series(np.dot(X.transpose(), temp))
        m = np.size(y)

        # #Cost function in vectorized form
        # h = np.dot(X,theta)
        J = float((1./(2*m)) * temp.T.dot(temp))  
        return J

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """

        #Computing the cost function for each theta combination
        # zs = np.array(  [self.costfunction(X, y_noise.reshape(-1,1),np.array([t0,t1]).reshape(-1,1)) 
                            # for t0, t1 in zip(np.ravel(T0), np.ravel(T1)) ] )

        # Z = zs.reshape(T0.shape)

        # theta range
        theta0_vals = np.linspace(-2,9,100)
        theta1_vals = np.linspace(-2,5,100)
        # print(theta0_vals,theta1_vals)
        J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

        # compute cost for each combination of theta
        c1=0; c2=0
        for i in theta0_vals:
            for j in theta1_vals:
                # print(i,j)
                t = np.array([i, j])
                temp = self.costfunction(X, y, t)
                # print(temp)
                J_vals[c1][c2] = temp
                c2=c2+1
            c1=c1+1
            c2=0

        theta0_history = []
        theta1_history = []
        cost_history = []
        for i in range(0,int(self.num_of_iterations/5),5):
            theta0_history.append(self.theta_history[i][0])
            theta1_history.append(self.theta_history[i][1])
            cost_history.append(self.cost_func_history[i])

            # print(self.cost_func_history)

            print(len(theta1_history), len(theta0_history), len(self.cost_func_history))

            #Setup of meshgrid of theta values
            T0, T1 = np.meshgrid(theta0_vals,theta1_vals)

            fig = plt.figure(figsize = (10,8))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.plot_surface(T0, T1, J_vals, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
            # ax = plt.axes(projection='3d')
            # ax.plot3D(theta0_history,theta1_history,self.cost_func_history)
            ax.plot(theta0_history,theta1_history,cost_history, marker = '*', color = 'r', alpha = .4, label = 'Gradient descent')
            ax.set_xlabel('theta 0')
            ax.set_ylabel('theta 1')
            ax.set_zlabel('Cost function')
            ax.view_init(45, 45)

            anglesx = np.array(theta0_history)[1:] - np.array(theta0_history)[:-1]
            anglesy = np.array(theta1_history)[1:] - np.array(theta1_history)[:-1]

            ax = fig.add_subplot(1, 2, 2)
            ax.contour(T0, T1, J_vals, 70, cmap = 'jet')
            ax.quiver(theta0_history[:-1], theta1_history[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .9)
            ax.set_xlabel('theta 0')
            ax.set_ylabel('theta 1')
            plt.savefig("./figures/scplots/"+str(i)+".png",dpi=400)


    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """

        X = list(np.array(X))
        y = list(y)
        x = []
        for i in range (len(X)):
            x.append(X[i][0])

        regression_line = []
        for i in x:
            regression_line.append((t_0*i)+t_1)
        
        for j in range (0,self.num_of_iterations,10):
            c,m = self.theta_history[j]
            regression_line = []
            for i in x:
                regression_line.append((m*i)+c)
            plt.figure()
            plt.scatter(x,y,color='b')
            plt.xlabel("x")
            plt.ylabel("y")
            m = float("{0:.2f}".format(m))
            c = float("{0:.2f}".format(c))
            plt.plot(x,regression_line,'-r')
            plt.title("m = "+str(m)+' and '+"c = "+str(c))
            # print(j)
            plt.savefig('./figures/line_fit/'+str(j+1)+'.png')

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        # theta range
        theta0_vals = np.linspace(-10,10,100)
        theta1_vals = np.linspace(-10,10,100)
        # print(theta0_vals,theta1_vals)
        J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

        # compute cost for each combination of theta
        c1=0; c2=0
        for i in theta0_vals:
            for j in theta1_vals:
                # print(i,j)
                t = np.array([i, j])
                temp = self.costfunction(X, y, t)
                # print(temp)
                J_vals[c1][c2] = temp
                c2=c2+1
            c1=c1+1
            c2=0

        theta0_history = []
        theta1_history = []
        for i in range(self.num_of_iterations):
            theta0_history.append(self.theta_history[i][0])
            theta1_history.append(self.theta_history[i][1])

        # print(self.cost_func_history)

        print(len(theta1_history), len(theta0_history), len(self.cost_func_history))
        
        anglesx = np.array(theta0_history)[1:] - np.array(theta0_history)[:-1]
        anglesy = np.array(theta1_history)[1:] - np.array(theta1_history)[:-1]

        #Setup of meshgrid of theta values
        T0, T1 = np.meshgrid(theta0_vals,theta1_vals)

        fig = plt.figure(figsize = (8,8))

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlabel('theta 0')
        ax.set_ylabel('theta 1')
        ax.contour(T0, T1, J_vals, 70, cmap = 'jet')
        # ax.quiver(T0,T1,J_vals,theta0_history[:-1], theta1_history[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .9)

        plt.show()
