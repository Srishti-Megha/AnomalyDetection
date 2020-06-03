# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:33:13 2020

@author: Srishti

library for baldwin data
"""
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import scipy as sp

# null values
def proportion_null_values(df):
    df_null = pd.DataFrame({'NonNull_Count': df.notnull().sum(), 'Null-Count': df.isnull().sum(), 'Null_Proportion': (df.isnull().sum()*100/df.shape[0]).round(2), 
                       'Null_Proportion_after_dropoing_rows_with_all_null': (df.dropna(how = 'all',axis = 0).isnull().sum()*100/df.shape[0]).round(2)})
    return df_null


### Descriptive statistics for Numeric Variables
def summary_stats(df):
    summary_stats = df.describe(percentiles = [.05, .25, .50, .75, .95], include = np.number)
    summary_stats.loc['skewness'] = df.skew(axis = 0, numeric_only = True)
    # round to two decimal places in python pandas
    summary_stats = summary_stats.round(2)
    
    return summary_stats
    
########### transform data ###############
#  z scaling
def zscale(df):
    scaler = StandardScaler()
    scaler.fit(df)
    df_zscale = pd.DataFrame(scaler.transform(df), columns = df.columns, index = df.index)
    return df_zscale



# get VIF
def get_vif(df, list_of_selected_features):
    X = sm.add_constant(df[list_of_selected_features])
    # Compute and view VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

#eleminate using vif
def eliminate_feature_using_VIF(df, list_of_selected_features, thres):
    vif = get_vif(df, list_of_selected_features)
    # new list of selected features
    new_list_of_selected_features = list_of_selected_features[:]
    while (vif[vif['variables']!='const']['VIF'] >= thres).any():
        new_list_of_selected_features.remove(vif[vif['variables']!='const'].sort_values('VIF', ascending=False).iloc[0,0])
        vif = get_vif(df, new_list_of_selected_features)
        
    # if elements get dropped based on VIF
    if set(list_of_selected_features) - set(new_list_of_selected_features):
        print("Dropped columns due to VIF:\n {}".format(set(list_of_selected_features) - set(new_list_of_selected_features)))
    else:
        print("All columns are kept")
        
    return new_list_of_selected_features





########### Missing Value Imputing Methods
#### KNN Impute #####
def KNNImpute(df, neighbors, weight="uniform"):
    imputer = KNNImputer(n_neighbors=neighbors, weights=weight)
    df_KNN_impute = pd.DataFrame(imputer.fit_transform(df), columns = df.columns, index = df.index)
    return df_KNN_impute
    

################### PCA ##############
#pca = PCA(n_components=2)
def optimum_component_for_PCA(df_scaled):
    pca = PCA().fit(df_scaled)
    fig, ax = plt.subplots(figsize=(12,8))
    xi = np.arange(1, df_scaled.shape[1]+1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, df_scaled.shape[1], step=1)) #change from 0-based array index to 1-based human-readable label
    plt.yticks(np.arange(0, 1.1, step=0.05))
    plt.ylabel('Cumulative variance (%)')
    plt.grid('on', axis='y' )
    plt.title('The number of components needed to explain variance')

    plt.show()
    return

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

#### Calculate Principal Components and malanobis Distance of Data#
def PCAMahalanobisDistanceOfData(df, num_of_components):
    # transform data to lower dimension
    pca = PCA(n_components=num_of_components, svd_solver= 'full')
    df_PCA = pd.DataFrame(pca.fit_transform(df), index = df.index)
    
    # calculate Mahalanobis Distance
    df_PCA['mahala'] = mahalanobis(x=df_PCA, data=df_PCA)
    
    return df_PCA
