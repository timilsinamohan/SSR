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
#red=3;white=1;rs
def innerfold_svr(x_test, y_test,x_train,y_train):
    svr_rbf = LinearSVR(random_state=1)
    svr_rbf.fit(x_train,y_train)
    pred_y = svr_rbf.predict(x_test)
    mse = mean_squared_error(y_test,pred_y)
    rmse = math.sqrt(mse)
    print rmse
    return rmse
    

if __name__ == '__main__':
    df = pd.read_csv("data/winequality-white.csv",sep = ";")
    #print df.shape
    data = df.iloc[:, :-1]
    data = data.iloc[:, :-1]
    data_feature = data.values
    n = data_feature.shape[0]
    print "Number of wines:",n
    get_true_value = df["alcohol"].values

    x = data_feature
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
        #print "this is train",x[train],y[train]
        #print "this is test",x[test],y[test]
        rmse_test[cnt] = innerfold_svr(X_Test, Y_Test,X_Train,Y_Train)
        cnt+=1

    print('RMSE Test Mean / Std: %f / %f' % (rmse_test.mean(), rmse_test.std()))


