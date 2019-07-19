__author__ = 'mohan'
from scipy import sparse
#from gbssl import LGC,HMN,PARW,OMNI,CAMLP
import networkx as nx
import random
import numpy as np
from sklearn import datasets
import time
from sklearn.neighbors import kneighbors_graph

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from scipy.io import loadmat
from sklearn import metrics
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import pairwise

from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import lil_matrix
from scipy import linalg
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
import math
import warnings
warnings.filterwarnings("ignore")
from scipy import sparse

def get_graph_matrix(X,training_nodes):

    GF = pairwise.rbf_kernel(X,X)
    GF = sparse.csr_matrix(GF)
    return GF



def get_lgc_scores(G,train_node_id,test_node_id,alpha):

    """ LGC computes the normalized Laplacian as its propagation matrix"""
    n = G.shape[0]
    graph = G.copy()
    degrees = graph.sum(axis=0).A[0]
    degrees[degrees==0] += 1  # Avoid division by 0
    D2 = np.sqrt(sparse.diags((1.0/degrees),offsets=0))
    S = D2.dot(graph).dot(D2)
    S = S*alpha

    ###create Base matrix#######

    all_values = get_true_value.copy()
    all_values[test_node_id] = 0
    state_vector = all_values.copy()


    ###Propagate the scores####
    remaining_iter = 100
    state = state_vector.copy()

    Base = state_vector
    while remaining_iter > 0:
        state = S.dot(state) + Base*(1-alpha)
        remaining_iter -= 1

    state = np.round_(state, decimals=2)
    return state

def get_harmonic_scores(G,train_node_id,test_node_id,yl):

    ##create Propagation Matrix###

    n = G.shape[0]
    graph = G.copy()
    degrees = graph.sum(axis=0).A[0]
    degrees[degrees==0] += 1  # Avoid division by 0
    D = sparse.diags((1.0/degrees),offsets=0)
    P = D.dot(graph).tolil()
    P[train_node_id] = 0
    all_values = np.zeros(n)
    all_values[train_node_id] = yl

    all_values[test_node_id] = 0
    state_vector = all_values.copy()
    #print "dekhau ta state_vector:",state_vector[0:10]

    ###Propagate the scores####
    remaining_iter = 30
    state = state_vector.copy()
    Base = state.copy()
    P = P.A

    while remaining_iter > 0:
        state = P.dot(state) + Base
        remaining_iter -= 1

    state = np.round_(state, decimals=2)


    return state

def get_hd_scores(G,train_node_id,test_node_id,alpha):
    n = G.shape[0]
    graph = G.copy()
    all_values = get_true_value.copy()
    all_values[test_node_id] = 0
    state_vector = all_values.copy()

#     ###Propagate the scores####
    remaining_iter = 30
    state = state_vector.copy()
    state = state.reshape(n,1)
    I = np.eye(n,n,dtype=np.float64)
    L = sparse.csgraph.laplacian(graph,normed=True)
    V = I + (-alpha/remaining_iter) * L

    while remaining_iter > 0:
        state = V.dot(state)
        remaining_iter -= 1

    return state

def get_bhd_scores(G,train_node_id,test_node_id,alpha):
    state_hmn = get_harmonic_scores(G,train_node_id,test_node_id,1)
    graph = G.copy()
    all_values = get_true_value.copy()
    all_values[test_node_id] = np.mean(all_values[train_node_id])
    remaining_iter = 100

    I = np.eye(n,n,dtype=np.float64)
    C = all_values - state_hmn
    state = C.copy()
    state = state.reshape(n,1)
    L = sparse.csgraph.laplacian(graph,normed=True)
    V = I + (-alpha/remaining_iter) * L

    while remaining_iter > 0:
        state = V.dot(state)
        remaining_iter -= 1

    state = state_hmn.reshape(n,1) + state

    return state

def innerfold(test_nodes, train_nodes):
    true_values = get_true_value.copy()
    graph_data = get_graph_matrix(data_feature,train_nodes)
    trained_labels_values = true_values[train_nodes]
    #predicted_values = get_harmonic_scores(graph_data,train_nodes,test_nodes,trained_labels_values)
    #predicted_values = get_hd_scores(graph_data,train_nodes,test_nodes,alpha = 1.0)
    #predicted_values = get_bhd_scores(graph_data,train_nodes,test_nodes,alpha = 1.0)
    predicted_values = get_lgc_scores(graph_data,train_nodes,test_nodes,alpha = 0.99)
    mse = mean_squared_error(true_values[test_nodes],predicted_values[test_nodes])
    rmse = math.sqrt(mse)
    print rmse
    #print accuracy
    return rmse

if __name__ == '__main__':

    df = pd.read_csv("data/airfoil_self_noise.dat",sep = "\t",header=None)
    data = df.iloc[:, :-1]
    data_feature = data.values
    n = data_feature.shape[0]
    get_true_value = df[5].values

    x = np.arange(n)
    y = get_true_value.copy()
    FOLDS = 10
    kfold = KFold(FOLDS, True, 1)
    cnt = 0
    rmse_test = np.zeros(FOLDS)
    for test, train in kfold.split(x,y):
        test_nodes = x[test]
        train_nodes = x[train]
        rmse_test[cnt] = innerfold(test_nodes, train_nodes)
        cnt+=1

    print('RMSE Test Mean / Std: %f / %f' % (rmse_test.mean(), rmse_test.std()))


