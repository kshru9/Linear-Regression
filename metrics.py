import math
import numpy as np
import pandas as pd

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    if (isinstance(y_hat,pd.Series)):
        y_hat = y_hat.tolist()
    if (isinstance(y,pd.Series)):
        y = y.tolist()
    
    l = len(y_hat)
    count = 0
    for i in range(l):
        if (y_hat[i] == y[i]):
            count+=1
    
    ans = (count/l)*100
    return ans

def precision(y_hat, y, cls):
    """
    Function to calculate the precision
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    if (isinstance(y_hat,pd.Series)):
        y_hat = y_hat.tolist()
    if (isinstance(y,pd.Series)):
        y = y.tolist()
    
    chosen_class = cls
    total_samples = len(y)

    pred_class_total = y_hat.count(chosen_class)
    
    correct_pred_count = 0
    for i in range(total_samples):
        if (y_hat[i] == chosen_class):
            if (y_hat[i] == y[i]):
                correct_pred_count+=1
    if (pred_class_total == 0):
        return None
    ans = (correct_pred_count/pred_class_total)*100
    return ans

def recall(y_hat, y, cls):
    """
    Function to calculate the recall
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()

    chosen_class = cls
    total_samples = len(y)

    total_samples_chosen = y.count(chosen_class)
    recall_count = 0

    for i in range(total_samples):
        if (y[i]==chosen_class):
            if (y_hat[i]==y[i]):
                recall_count+=1

    ans = (recall_count/total_samples_chosen)*100
    return ans

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    diff = [0]*len(y)
    for i in range (len(diff)):
        diff[i]=(y_hat[i]-y[i])**2
    ans = sum(diff)/len(y)
    ans = math.sqrt(ans)
    return ans

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    diff = [0]*len(y)
    for i in range (len(diff)):
        diff[i]=abs(y_hat[i]-y[i])
    return sum(diff)/len(y)