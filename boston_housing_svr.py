__author__ = 'mohan'
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR
import math
import warnings
warnings.filterwarnings("ignore")

def innerfold_svr(x_test, y_test,x_train,y_train):
    svr = LinearSVR(random_state=2)
    svr.fit(x_train,y_train)
    pred_y = svr.predict(x_test)
    mse = mean_squared_error(y_test,pred_y)
    rmse = math.sqrt(mse)
    print rmse
    return rmse
    

if __name__ == '__main__':
    
    data = datasets.load_boston()
    data_feature = data.data
    
    x = data_feature
    get_true_value = np.copy(data.target)

    y = get_true_value.copy()
    FOLDS = 10
    kfold = KFold(FOLDS, True, random_state=1)
    cnt = 0
    rmse_test = np.zeros(FOLDS)
    for test, train in kfold.split(x,y):
        X_Test = x[test]
        Y_Test = y[test]
        X_Train = x[train]
        Y_Train = y[train]
        #print "this is train",x[train],y[train]
        #print "this is test",x[test],y[test]
        rmse_test[cnt] = innerfold_svr(X_Test, Y_Test,X_Train,Y_Train)
        cnt+=1

    print('RMSE Test Mean / Std: %f / %f' % (rmse_test.mean(), rmse_test.std()))


