#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:43:45 2018

@author: debanjana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


epoch=100

def grad_Descent_train(alpha,theta,given,X):

    predicted=np.sum(X*theta,axis=1) #linear combination of features
    error=(predicted-given)
    error = np.reshape(error, (error.shape[0], 1)) #converting error into a column vector
#   print(error)
    grad=sum((alpha/len(given))*(X*error)) #one element of error * one column of X
    theta=(theta-grad)
    mse_linear= np.sum(error**2)/(2*len(X)) #mean squared error #for training
    #rmse=np.sqrt(2*mse_linear)    #root men squared #for test
    print(theta, mse_linear)
    return theta

def grad_Descent_test(theta,given,X,alpha,msg):

   predicted=np.sum(X*theta,axis=1) #theta is considered a column vector of shape(n,) #sum across rows
   error=(predicted-given)
   error = np.reshape(error, (error.shape[0], 1)) #converting error into a column vector
   mse= np.sum(error**2)/(2*len(X)) #mean squared error #for training
   rmse=np.sqrt(2*mse)    #root men squared #for test
   print("For test :",theta, rmse)
   #denormalizing
   de_predicted=(predicted*std_deviation)+mean
   print(de_predicted)
   if alpha==0.02 and msg=="LINEAR":
       np.savetxt("test_prediction_comb_of_features_linear.csv",de_predicted)
   elif alpha==0.02 and msg=="QUAD":
       np.savetxt("test_prediction_comb_of_features_quadratic.csv",de_predicted)
   elif alpha==0.02 and msg=="CUBIC":
       np.savetxt("test_prediction_comb_of_features_cubic.csv",de_predicted)
   return rmse

df=pd.read_csv("C:/Users/Debanjana/Desktop/MLAssignment1/kc_house_data.csv")
list(df)
X=df[["sqft","floors","bedrooms","bathrooms"]] #don't add bias before normalization #mean=1 & std=0 for bias col
Y=df["price"]
Y_array=Y.values
X_array=X.values

#cubic features
X_cube=np.array([X_array[:,0]**3 , 3*X_array[:,0]**2*X_array[:,1] , 3*X_array[:,0]**2*X_array[:,2] , 3*X_array[:,0]**2*X_array[:,3] , 3*X_array[:,0]*X_array[:,1]**2 , 6*X_array[:,0]*X_array[:,1]*X_array[:,2] , 6*X_array[:,0]*X_array[:,1]*X_array[:,3] , 3*X_array[:,0]*X_array[:,2]**2 , 6*X_array[:,0]*X_array[:,2]*X_array[:,3] , 3*X_array[:,0]*X_array[:,3]**2 , X_array[:,1]**3 , 3*X_array[:,1]**2*X_array[:,2] , 3*X_array[:,1]**2*X_array[:,3] , 3*X_array[:,1]*X_array[:,2]**2 , 6*X_array[:,1]*X_array[:,2]*X_array[:,3] , 3*X_array[:,1]*X_array[:,3]**2 , X_array[:,2]**3 , 3*X_array[:,2]**2*X_array[:,3] , 3*X_array[:,2]*X_array[:,3]**2 , X_array[:,3]**3])
#quadratic features
X_quad=np.array([X_array[:,0]**2 , 2*X_array[:,0]*X_array[:,1] , 2*X_array[:,0]*X_array[:,2] , 2*X_array[:,0]*X_array[:,3] , X_array[:,1]**2 , 2*X_array[:,1]*X_array[:,2] , 2*X_array[:,1]*X_array[:,3] , X_array[:,2]**2 , 2*X_array[:,2]*X_array[:,3] , X_array[:,3]**2])

X_cube=X_cube.T
X_quad=X_quad.T
X_cube = np.hstack((X_array,X_quad,X_cube)) #combination of linear ,quad & cube
X_quad = np.hstack((X_array,X_quad)) #combination of linear & quad

print("cube:",X_cube)
print("quad",X_quad)
print("----------------------------------------------")

#normalization
mean=np.mean(Y_array,axis=0)
std_deviation=np.std(Y_array,axis=0)
X_array=(X_array-np.mean(X_array,axis=0))/np.std(X_array,axis=0)
Y_array=(Y_array-mean)/std_deviation
X_quad=(X_quad-np.mean(X_quad,axis=0))/np.std(X_quad,axis=0)
X_cube=(X_cube-np.mean(X_cube,axis=0))/np.std(X_cube,axis=0)

print("cube:",X_cube)
print("quad",X_quad)
print("----------------------------------------------")
 #combination of linear & cube
bias= np.ones(len(X_array))
bias= np.reshape(bias, (len(bias),1))

X_array = np.hstack((bias, X_array))
X_quad = np.hstack((bias, X_quad))
X_cube = np.hstack((bias, X_cube))

part=int(.8*len(df))
train_X=X_array[:part]
train_X_quad=X_quad[:part]
train_X_cube=X_cube[:part]
train_Y=Y_array[:part]
test_X=X_array[part:]
test_X_quad=X_quad[part:]
test_X_cube=X_cube[part:]
test_Y=Y_array[part:]

#normalize
#print(X_array)
alpha=[0,0.01,0.02,0.03,0.04,0.05,0.1]

rmse_linear=np.zeros(len(alpha))
rmse_quad=np.zeros(len(alpha))
rmse_cubic=np.zeros(len(alpha))
#rmse=np.zeros(epoch)
for k,j in enumerate(alpha):
    print("For alpha ----------------------------------------- :",j)
    theta_linear=np.array([random.uniform(0, 1) for i in range(len(train_X[0]))])
    theta_quad=np.array([random.uniform(0, 1) for i in range(len(train_X_quad[0]))])
    theta_cubic=np.array([random.uniform(0, 1) for i in range(len(train_X_cube[0]))])
    for i in range(epoch):
        print("For linear mse :")
        theta_linear=grad_Descent_train(j,theta_linear,train_Y,train_X)
        print("For quadratic mse:")
        theta_quad=grad_Descent_train(j,theta_quad,train_Y,train_X_quad)
        print("For cube mse:")
        theta_cubic=grad_Descent_train(j,theta_cubic,train_Y,train_X_cube)
    print("For linear rmse:")
    rmse_linear[k]=grad_Descent_test(theta_linear,test_Y,test_X,j,"LINEAR")
    print("For quadratic rmse :")
    rmse_quad[k]=grad_Descent_test(theta_quad,test_Y,test_X_quad,j,"QUAD")
    print("For cube rmse :")
    rmse_cubic[k]=grad_Descent_test(theta_cubic,test_Y,test_X_cube,j,"CUBIC")

#plotting the errors    
f=plt.figure()
plt.plot(alpha,rmse_linear,color="red")
plt.ylabel("root mean square error")
plt.xlabel("learning rate")
plt.title("Linear RMSE vs. Learning Rate")
f.show()

g=plt.figure()
plt.plot(alpha,rmse_quad,color="green")
plt.ylabel("root mean square error")
plt.xlabel("learning rate")
plt.title("Quadratic RMSE vs. Learning Rate")
g.show()

h=plt.figure()
plt.plot(alpha,rmse_cubic,color="blue")
plt.ylabel("root mean square error")
plt.xlabel("learning rate")
plt.title("Cubic RMSE vs. Learning Rate")
h.show()
