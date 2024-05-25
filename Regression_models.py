#!/usr/bin/env python
# coding: utf-8

# # 1.1 a)
#  
#  Import Libraries

# In[2]:


import numpy as np# numpy is library
import pandas as pd# pandas is library
import matplotlib.pyplot as plt# here matplotlib is library and pyplot is module


# Import dataset

# In[3]:


bike_data= pd.read_csv("/Users/satyarth/Downloads/bike_share_.csv")# declare variable and store the data set
# here read_csv is function from pandas library
bike_data


# In[4]:


bike_data_temp_and_cnt=bike_data[["temp","cnt"]]
bike_data_temp_and_cnt


# # 1.1(b) 
#  
#  scatter plot between temp and cnt

# In[5]:


bike_data_temp_and_cnt.plot(x='temp',y='cnt',kind='scatter')


# # 1.1(c)
# 
# prediction of model

# In[6]:


X_trn= bike_data_temp_and_cnt['temp'].values
X_trn# Feature variable
X_trn.shape


# In[20]:


Y_trn=bike_data_temp_and_cnt['cnt'].values# we added values keyword ,we want data in numpy array
Y_trn#response variable


# Split data set into training and testing data

# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_trn, Y_trn, test_size=0.3, random_state = 1)


# In[29]:


X_trains=X_train.reshape(-1,1)# we have used reshape function here because we have single feature i.e."temp".
X_trains
# please note-Reshape your data either using array.reshape(-1, 1) if your data has a
#single feature or array.reshape(1, -1) if it contains a single sample.


# In[30]:


Y_trains =Y_train.reshape(-1,1)
Y_trains


# Traing the simple linear regression model on training set

# predicting the test set result

# In[32]:


X_tests= X_test.reshape(-1,1)
Y_tests=Y_test.reshape(-1,1)
Y_tests.shape


# In[117]:


from sklearn.linear_model import LinearRegression
#here sklearn is library and linear_model is module which contains LinearRegression class
regressor= LinearRegression()# here we have created object named'regressor' of class LinearRegression.

#here model is ready to use
regressor.fit(X_trains,Y_trains)# will train model based on training set


# In[116]:


Y_predict_train=regressor.predict(X_trains)


# In[34]:


Y_predict_test=regressor.predict(X_tests)
Y_predict_test.shape


# # 1.1(d)
# 
# draw the regression line with scatter plot

# In[118]:


plt.scatter(X_trains, Y_trains,color='red')
plt.plot(X_trains, regressor.predict(X_trains),color='blue')
plt.xlabel("temp")
plt.ylabel("cnt")
plt.show


# In[40]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_tests, Y_predict_test)
r_squared = r2_score(Y_tests, Y_predict_test)
r_squared


# # 1.1(e)
# 
# calculate R squared

# In[38]:


r_squared_train =r2_score(Y_train,Y_predict_train)


# In[39]:


r_squared_train


# Visualize the test set result based on training data set model

# In[113]:


plt.scatter(X_tests, Y_tests,color='red')
plt.plot(X_trains, regressor.predict(X_trains),color='blue')
plt.xlabel("temp")
plt.ylabel("cnt")
plt.show


# # Multiple linear regression model
# 
# 1.2 a) 
# please note ,I have splited data into training and testing set .So please review based on this

# In[41]:


multiple_linear_regression=bike_data.iloc[:,[9,10,11,12,15]]
multiple_linear_regression



# In[42]:


feature_columns = multiple_linear_regression.drop(['cnt'], axis=1)
feature_columns


# In[43]:


X_multiple=feature_columns
X_multiple


# In[44]:


Y_multiple=multiple_linear_regression.cnt
Y_multiple


# Split data into training and testing set

# In[45]:


X_train_multiple,X_test_multiple,Y_train_multiple,Y_test_multiple = train_test_split(X_multiple,Y_multiple,test_size=0.3,random_state=1)


# In[46]:


X_train_multiple


# In[47]:


Y_train_multiple


# # 1.2(b)
# 
# train the model  (I have splitted data into training and testing)

# In[48]:


multiple_regression_model=LinearRegression()


# In[49]:


multiple_regression_model.fit(X_train_multiple,Y_train_multiple)


# In[54]:


Y_multiple_predict=multiple_regression_model.predict(X_train_multiple)
Y_multiple_predict


# In[55]:


for idx, col_name in enumerate(X_train_multiple.columns):
    print("Coefficeint of {} is {}".format(col_name, multiple_regression_model.coef_[idx]))


# In[56]:


print("Intercept: ", multiple_regression_model.intercept_)


# # 1.2(c)
# 
# predict no of sales

# In[57]:


test_input =[[0.34416,0.363625,0.805833,0.16044]]
print(f'Predicted sales for bikes: {multiple_regression_model.predict(test_input)}')


# # 1.2(d)
# 
# calculate mean squared error

# In[58]:


from sklearn.metrics import mean_squared_error
mean_squared_error(Y_train_multiple,Y_multiple_predict)


# # 1.2(e)
# 
# plot the histrogram of residuals

# In[59]:


histrogram=np.subtract (Y_train_multiple,Y_multiple_predict)
histrogram


# In[60]:


plt.hist(histrogram)


# In[61]:


X_multiple_for_r_square, y_multiple_for_r_square = X_train_multiple[['temp','atemp','hum','windspeed']], Y_train_multiple
multiple_regression_model.score(X_multiple_for_r_square, y_multiple_for_r_square)


# In[62]:


adjusted_R_square=1 - (1-multiple_regression_model.score(X_multiple_for_r_square, y_multiple_for_r_square))*(len(y_multiple_for_r_square )-1)/(len(y_multiple_for_r_square )-X_multiple_for_r_square.shape[1]-1)
adjusted_R_square


# # Non-linear regression model

# 1.3 a- temp_2 variable

# In[75]:


variable=bike_data['temp']#stored 'temp' data set to variable
variable_new=variable.values#converted in array
variable_new_1 =variable_new.reshape(-1,1)
variable_new_1# will give bike data set



# In[65]:


variable_2= np.square(variable)
variable_2#created temp_2 variable


# # 1.3(b) 
# plot the model (This is one method to predict.I have used another method to predict model which is mentioned after this method)

# In[87]:


from sklearn.preprocessing import PolynomialFeatures# here sklearn is library , preprocessing is module 
#Under preprocessing module,polynomial feature is class 
non_linear_regressor=PolynomialFeatures(degree=2)# we want to create polynomial model of degree 2 , so here degree=2
# non_linear_regressor is object of PolynomialFeature class

x_non_linear= non_linear_regressor.fit_transform(variable_new_1)
# here we create matrix of variable_new_1 with variable_2(transformed from variable_new_1)


non_linear_regression=LinearRegression()# here we are building model
non_linear_regression.fit(x_non_linear,Y_multiple)# we called Y_multiple from multiple linear regression part
print("coefficients are:{}".format(non_linear_regression.coef_))
print("intercept is:{}".format(non_linear_regression.intercept_))


# # 1.3 (c)
# 
# plot the model(I have plotted in 2-d in this method but plotted in 3-d in second method)

# In[82]:


plt.scatter(variable_new_1,Y_multiple,color='red')
plt.plot(variable_new_1,non_linear_regression.predict(x_non_linear))
# here under predict(x_non_linear) ,we are getting predicted values of both variable_new_1 and variable_2
plt.xlabel('temp')
plt.ylabel('cnt')         
plt.show()         


# # 1.3 (d) 
# 
# calculate the r squared

# In[68]:


Y_predict_non_linear= non_linear_regression.predict(x_non_linear)


# In[69]:


r_squared_non_linear =r2_score(Y_multiple,Y_predict_non_linear)
r_squared_non_linear# iF we compare with R^2 of multiple linear model, R^2 for non linear explains little better the model.


# # Second method to calculate non linear model

# In[95]:


X_p,Y_p=np.vstack([variable_new,variable_2]).T,Y_multiple
non_linear_regressor_second= LinearRegression().fit(X_p,Y_p)
print("R_squared:{}".format(non_linear_regressor_second.score(X_p,Y_p)))
print("coefficients:{}".format(non_linear_regressor_second.coef_))
print("intercept:{}".format(non_linear_regressor_second.intercept_))


# In[101]:


x_grid, y_grid=np.meshgrid(X_p[:,0],X_p[:,1])
x_grid,y_grid


# In[102]:


z=non_linear_regressor_second.intercept_+x_grid*non_linear_regressor_second.coef_[0]+y_grid*non_linear_regressor_second.coef_[1]


# In[112]:


fig=plt.figure()
axes_in_three_D= plt.axes(projection='3d')
axes_in_three_D.plot_surface(x_grid,y_grid,z,rstride=1,cstride=1,alpha=0.3,color='r')
axes_in_three_D.scatter(X_p[:,0],X_p[:,1],Y_p,s=100)
axes_in_three_D.view_init(40,-30)


# In[ ]:




