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
    svr_rbf = LinearSVR(random_state=3)
    svr_rbf.fit(x_train,y_train)
    pred_y = svr_rbf.predict(x_test)
    mse = mean_squared_error(y_test,pred_y)
    rmse = math.sqrt(mse)
    print rmse
    return rmse
    

if __name__ == '__main__':
    
    file = "data/Advertising.csv"
    advert = pd.read_csv(file)
    X = advert["TV"].values
    Y = advert['sales'].values
    
    x = X.reshape(len(X), 1)
    #print x
    get_true_value = Y
    y = get_true_value.copy()
    FOLDS = 10
    kfold = KFold(FOLDS, True, 1)
    cnt = 0
    rmse_test = np.zeros(FOLDS)
    for test, train in kfold.split(x,y):
        X_Test = x[test]
        Y_Test = y[test]
        X_Train = x[train]
        Y_Train = y[train]

        rmse_test[cnt] = innerfold_svr(X_Test, Y_Test,X_Train,Y_Train)
        cnt+=1

    print('RMSE Test Mean / Std: %f / %f' % (rmse_test.mean(), rmse_test.std()))


