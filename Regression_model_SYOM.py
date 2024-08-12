#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 09:47:19 2024

@author: Satyarth Omar
# Author: Satyarth Omar

# Date of code update: July 14, 2024

# Purpose of the script: performing linear regression analysis on a data set, 
# and obtaining degree of linear reletionship b/w dependent and independent 
# variables 





# # Part A>>>> Linear regression model with results
#  Import Libraries, reading data, training model and plot visualization

# 1.1(a)imported libraries and loaded dataset.Extracted 'temp' and 'cnt' columns to perform operations.
# 
#         
# 1.1(b)Visualized the scatter plot between 2 feature vairables 'temp' and 'cnt'.
# 
#         
# 1.1(c)Built linear regression model , made predictions and calculated r-square.
# 
#         
# 1.1(d)Plotted the regression line.

# # 1.1(a)
"""


import numpy as np# numpy is library
import pandas as pd# pandas is library
#import matplotlib.pyplot as plt# here matplotlib is library and pyplot is module

def model_fit(model_type,X_fit_train,X_fit_test,Y_fit_train):
    if model_type == 'linear':
        from sklearn.linear_model import LinearRegression
        #here sklearn is library and linear_model is module which contains LinearRegression class
        regressor= LinearRegression()# 
        regressor.fit(X_fit_train,Y_fit_train)# will train model based on training set
        Y_predict_train=regressor.predict(X_fit_train)
        Y_predict_test=regressor.predict(X_fit_test)
    return Y_predict_train,Y_predict_test 

def calc_KPI(y_data,y_model):
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_data,y_model)
    rsq = r2_score(y_data,y_model)
    return mse,rsq
    
    
# Import dataset and splitting into training and test set
bike_data= pd.read_csv("/Users/sidomar/Documents/git_collaboration/Regression-/data/bike_share_.csv")# declare variable and store the data set
bike_data_temp_and_cnt=bike_data[["temp","cnt"]]
bike_data_temp_and_cnt

X_trn= bike_data_temp_and_cnt['temp'].values
Y_trn= bike_data_temp_and_cnt['cnt'].values

# splitting data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_trn, Y_trn, test_size=0.3, random_state = 1)

# please note-Reshape your data either using array.reshape(-1, 1) if your data has a
#single feature or array.reshape(1, -1) if it contains a single sample.

X_trains=X_train.reshape(-1,1)# we have used reshape function here because we have single feature i.e."temp".
Y_trains =Y_train.reshape(-1,1)
X_tests= X_test.reshape(-1,1)
Y_tests=Y_test.reshape(-1,1)

[Y_predict_train,Y_predict_test] = model_fit('linear',X_trains,X_tests,Y_trains)
# training model and evaluating performance

 ### KPI calculation 

mse = calc_KPI(Y_tests, Y_predict_test)[0]
r_squared = calc_KPI(Y_tests, Y_predict_test)[1]
r_squared_train = calc_KPI(Y_train,Y_predict_train)[1]

#multi linear regression model
multiple_linear_regression=bike_data.iloc[:,[9,10,11,12,15]]
X_multiple = multiple_linear_regression.drop(['cnt'], axis=1)
Y_multiple=multiple_linear_regression.cnt
X_train_multiple,X_test_multiple,Y_train_multiple,Y_test_multiple = train_test_split(X_multiple,Y_multiple,test_size=0.3,random_state=1)
[Y_multiple_predict_train,Y_multiple_predict_test] = model_fit('linear',X_train_multiple,X_test_multiple,Y_train_multiple)

#X_multiple_for_r_square, y_multiple_for_r_square = X_train_multiple[['temp','atemp','hum','windspeed']], Y_train_multiple
#multiple_regression_model.score(X_multiple_for_r_square, y_multiple_for_r_square)


mse_mult = calc_KPI(Y_train_multiple,Y_multiple_predict_train)
#adjusted_R_square=1 - (1-multiple_regression_model.score(X_multiple_for_r_square, y_multiple_for_r_square))*(len(y_multiple_for_r_square )-1)/(len(y_multiple_for_r_square )-X_multiple_for_r_square.shape[1]-1)

variable=bike_data['temp']#stored 'temp' data set to variable
variable_new=variable.values#converted in array
variable_new_1 =variable_new.reshape(-1,1)
variable_2= np.square(variable)


from sklearn.preprocessing import PolynomialFeatures# here sklearn is library , preprocessing is module 
#Under preprocessing module,polynomial feature is class 
non_linear_regressor=PolynomialFeatures(degree=2)# we want to create polynomial model of degree 2 , so here degree=2
# non_linear_regressor is object of PolynomialFeature class

x_non_linear= non_linear_regressor.fit_transform(variable_new_1)
# here we create matrix of variable_new_1 with variable_2(transformed from variable_new_1)

[Y_predict_non_linear,Y_predict_test_non_linear] = model_fit('linear',x_non_linear,x_non_linear,Y_multiple)


#Y_predict_non_linear= non_linear_regression.predict(x_non_linear)
r_squared_non_linear =calc_KPI(Y_multiple,Y_predict_non_linear)[1]
X_p,Y_p=np.vstack([variable_new,variable_2]).T,Y_multiple
non_linear_regressor_second= model_fit('linear',X_p,X_p,Y_p)
x_grid, y_grid=np.meshgrid(X_p[:,0],X_p[:,1])
x_grid,y_grid
#z=non_linear_regressor_second.intercept_+x_grid*non_linear_regressor_second.coef_[0]+y_grid*non_linear_regressor_second.coef_[1]


