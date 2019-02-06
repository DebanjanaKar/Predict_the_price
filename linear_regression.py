#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:29:41 2018

@author: debanjana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


alpha=0.05
epoch=100
#reg=0.5

def grad_Descent_train(alpha,theta,given,X,reg):

   predicted=np.sum(X*theta,axis=1) #theta is considered a column vector of shape(n,) #sum across rows
#   print("predicted")
#   print(predicted)
#   print(given)
   error=(predicted-given)
#   print("error")
   error = np.reshape(error, (error.shape[0], 1)) #converting error into a column vector
#   print(error)
   grad=sum((alpha/len(given))*(X*error)) #one element of error * one column of X
   reg_term=1-((alpha*reg)/len(given))
   theta=((theta*reg_term)-grad)
#   print("theta")
#   print(theta)
#   print(theta.shape)
   mse= np.sum(error**2)/(2*len(X))+(reg*np.sum(theta*theta)) #mean squared error #for training
   #rmse=np.sqrt(2*mse)    #root mean squared #for test
   #print(reg,theta, mse)
   return theta, mse

def grad_Descent_test(reg,theta,given,X):

   predicted=np.sum(X*theta,axis=1) #theta is considered a column vector of shape(n,) #sum across rows
   error=(predicted-given)
   error = np.reshape(error, (error.shape[0], 1)) #converting error into a column vector
   mse= np.sum(error**2)/(2*len(X)) #mean squared error #for training
   rmse=np.sqrt(2*mse)    #root mean squared #for test
   print(theta, rmse)
   #denormalizing
   de_predicted=(predicted*std_deviation)+mean
   print(de_predicted)
   if(reg==0.5):
       np.savetxt("test_prediction_linear_regression_0.5.csv",de_predicted)
   return rmse

df=pd.read_csv("C:/Users/Debanjana/Desktop/MLAssignment1/kc_house_data.csv")
list(df)
X=df[["sqft","floors","bedrooms","bathrooms"]]
Y=df["price"]
Y_array=Y.values
X_array=X.values
print(Y_array)

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
reg=np.array([0.0,0.2,0.5,0.6,0.8,1.0,1.5])

mse=np.zeros(epoch)
rmse=np.zeros(len(reg))

for k,j in enumerate(reg):
    theta=np.array([random.uniform(0, 1) for i in range(5)])#np.zeros(5))
    #theta=np.zeros(5)
    for i in range(epoch):
        theta,mse[i]=grad_Descent_train(alpha,theta,train_Y,train_X,j)
    print("----------------------------%s--------------------------------",j)
    rmse[k]=grad_Descent_test(j,theta,test_Y,test_X)
    plt.plot(mse)
    plt.ylabel("mean squared error on train")
    plt.xlabel("# epochs")
    plt.show()

f=plt.figure()
plt.xlabel("regularization values")
plt.ylabel("root mean squared error")
plt.plot(reg,rmse)
plt.title("Variation of test RMSE with the weightage of the regularization terms")
f.show()