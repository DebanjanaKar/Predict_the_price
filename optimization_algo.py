#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:37:09 2018

@author: debanjana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


alpha=0.05
epoch=100


def grad_Descent_train(alpha,theta,given,X):

   predicted=np.sum(X*theta,axis=1) #theta is considered a column vector of shape(n,) #sum across rows
   error=(predicted-given)
   error = np.reshape(error, (error.shape[0], 1)) #converting error into a column vector
   #print(X*error)
   grad=sum((alpha/len(given))*(X*error)) #one element of error * one column of X
   theta=(theta-grad)
   mse= np.sum(error**2)/(2*len(X)) #mean squared error #for training
   return theta,mse

def irls_train(alpha,theta,given,X):

   predicted=np.sum(X*theta,axis=1) #theta is considered a column vector of shape(n,) #sum across rows
   error=(predicted-given)
   error = np.reshape(error, (error.shape[0], 1)) #converting error into a column vector
   grad=sum(X*error)
   print(grad)
   hessian=sum(X*X)
   print(hessian)
   theta=theta-(grad/hessian)
   mse= np.sum(error**2)/(2*len(X)) #mean squared error #for training
   return theta,mse
           
           
def test(theta,given,X,alpha,msg):

   predicted=np.sum(X*theta,axis=1) #theta is considered a column vector of shape(n,) #sum across rows
   error=(predicted-given)
   error = np.reshape(error, (error.shape[0], 1)) #converting error into a column vector
   mse= np.sum(error**2)/(2*len(X)) #mean squared error #for training
   rmse=np.sqrt(2*mse)    #root men squared #for test
   print(theta, rmse)
   #denormalizing
   de_predicted=(predicted*std_deviation)+mean
   if alpha==0.02 and msg=="GD" :
       np.savetxt("test_prediction_optimization_algo_GradDescent.csv",de_predicted)
   elif alpha==0.02 and msg=="IRLS" :
       np.savetxt("test_prediction_optimization_algo_IRLS.csv",de_predicted)
   return rmse

df=pd.read_csv("C:/Users/Debanjana/Desktop/MLAssignment1/kc_house_data.csv")
list(df)
X=df[["sqft","floors","bedrooms","bathrooms"]] 
Y=df["price"]
Y_array=Y.values
X_array=X.values
#normalization
mean=np.mean(Y_array,axis=0)
std_deviation=np.std(Y_array,axis=0)
X_array=(X_array-np.mean(X_array,axis=0))/np.std(X_array,axis=0)
Y_array=(Y_array-mean)/std_deviation


bias= np.ones(len(X_array))
bias= np.reshape(bias, (len(bias),1))

X_array = np.hstack((bias, X_array))

part=int(.8*len(df))
train_X=X_array[:part]
train_Y=Y_array[:part]
test_X=X_array[part:]
test_Y=Y_array[part:]

#normalize
#print(X_array)
alpha=[0,0.01,0.02,0.03,0.04,0.05,0.1]

rmse_grad=np.zeros(len(alpha))
rmse_irls=np.zeros(len(alpha))
mse_grad=np.zeros(epoch)
mse_irls=np.zeros(epoch)
#rmse=np.zeros(epoch)
for k,j in enumerate(alpha):
    print("For alpha ----------------------------------------- :",j)
    theta_grad=np.array([random.uniform(0, 1) for i in range(len(train_X[0]))])
    theta_irls=np.array([random.uniform(0, 1) for i in range(len(train_X[0]))])
    
    for i in range(epoch):
        print("For gradient descent :")
        theta_grad,mse_grad[i]=grad_Descent_train(j,theta_grad,train_Y,train_X)
        print("For iterative reweighted least square:")
        theta_irls,mse_irls[i]=irls_train(j,theta_irls,train_Y,train_X)
        
    rmse_grad[k]=test(theta_grad,test_Y,test_X,j,"GD")
    rmse_irls[k]=test(theta_irls,test_Y,test_X,j,"IRLS")


plt.plot(mse_grad,color="red")
plt.xlabel("#epoch")
plt.ylabel("mse")
plt.title("For Gradient Descent")
plt.show()

plt.plot(mse_irls)
plt.xlabel("#epoch")
plt.ylabel("mse")
plt.title("For Iterative Reweighted Least Square")
plt.show()
    
f=plt.figure()
plt.plot(alpha,rmse_grad,color="red")
plt.xlabel("learning rate")
plt.ylabel("rmse")
plt.title("For Gradient Descent")
f.show()

f=plt.figure()
plt.plot(alpha,rmse_irls)
plt.xlabel("learning rate")
plt.ylabel("rmse")
plt.title("For IRLS")
f.show()

