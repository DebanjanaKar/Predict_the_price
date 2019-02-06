#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:45:45 2018

@author: debanjana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


alpha=0.05
epoch=100
#reg=0.5


def grad_Descent_train_mse(alpha,theta,given,X):

   predicted=np.sum(X*theta,axis=1) #theta is considered a column vector of shape(n,) #sum across rows
   error=(predicted-given)
   error = np.reshape(error, (error.shape[0], 1)) #converting error into a column vector
#   print(error)
   grad=sum((alpha/len(given))*(X*error)) #one element of error * one column of X
   theta=(theta-grad)
   mse= np.sum(error**2)/(2*len(X)) #mean squared error #for training
   print(theta,mse)
   return theta

def grad_Descent_train_mce(alpha,theta,given,X):

   predicted=np.sum(X*theta,axis=1) #theta is considered a column vector of shape(n,) #sum across rows
   error=(predicted-given)
   error = np.reshape(error, (error.shape[0], 1)) #converting error into a column vector
#   print(error)
   grad=(alpha/len(given))*sum(X*(error*error)) #one element of error * one column of X
   theta=(theta-grad)
   mce= np.sum(error**3)/(3*len(X)) #mean squared error #for training
   print(theta,mce)
   return theta

def grad_Descent_train_ame(alpha,theta,given,X):

   predicted=np.sum(X*theta,axis=1)
   error=(predicted-given)
   error = np.reshape(error, (error.shape[0], 1)) 
#   print(error)
   grad=(alpha/len(given))*sum(X*(error/abs(error)))
   theta=(theta-grad)
   ame= np.sum(error/abs(error))/(len(X)) #mean squared error #for training
   print(theta,ame)
   return theta

def grad_Descent_test(theta,given,X,alpha,msg):

   predicted=np.sum(X*theta,axis=1) 
   error = np.reshape(error, (error.shape[0], 1)) 
   mse= np.sum(error**2)/(2*len(X))
   rmse=np.sqrt(2*mse)   
   print(theta, rmse)
   #denormalizing
   de_predicted=(predicted*std_deviation)+mean
   if alpha==0.02 and msg=="MSE":
       np.savetxt("test_prediction_cost_func_MSE.csv",de_predicted)
   elif alpha==0.02 and msg=="MCE" :
       np.savetxt("test_prediction_cost_func_MCE.csv",de_predicted)
   elif alpha==0.02 and msg=="AME" :
       np.savetxt("test_prediction_cost_func_AME.csv",de_predicted)
   return rmse

df=pd.read_csv("C:/Users/Debanjana/Desktop/MLAssignment1/kc_house_data.csv")
list(df)
#df["bias"]=np.ones(len(df))
X=df[["sqft","floors","bedrooms","bathrooms"]] #don't add bias before normalization #mean=1 & std=0 for bias col
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

rmse_mse=np.zeros(len(alpha))
rmse_mce=np.zeros(len(alpha))
rmse_ame=np.zeros(len(alpha))
#rmse=np.zeros(epoch)
for k,j in enumerate(alpha):
    print("For alpha ----------------------------------------- :",j)
    theta_mse=np.array([random.uniform(0, 1) for i in range(len(train_X[0]))])
    theta_mce=np.array([random.uniform(0, 1) for i in range(len(train_X[0]))])
    theta_ame=np.array([random.uniform(0, 1) for i in range(len(train_X[0]))])
    for i in range(epoch):
        print("For mse :")
        theta_mse=grad_Descent_train_mse(j,theta_mse,train_Y,train_X)
        print("For mce:")
        theta_mce=grad_Descent_train_mce(j,theta_mce,train_Y,train_X)
        print("For ame:")
        theta_ame=grad_Descent_train_ame(j,theta_ame,train_Y,train_X)
    print("For test rmse using mse:")
    rmse_mse[k]=grad_Descent_test(theta_mse,test_Y,test_X,j,"MSE")
    print("For test rmse using mce :")
    rmse_mce[k]=grad_Descent_test(theta_mce,test_Y,test_X,j,"MCE")
    print("For test rmse using ame :")
    rmse_ame[k]=grad_Descent_test(theta_ame,test_Y,test_X,j,"AME")


#plotting the errors    
f=plt.figure()
plt.plot(alpha,rmse_mse,color="red")
plt.ylabel("root mean square error")
plt.xlabel("learning rate")
plt.title("Cost function = Mean Square Error")
f.show()

g=plt.figure()
plt.plot(alpha,rmse_mce,color="green")
plt.ylabel("root mean square error")
plt.xlabel("learning rate")
plt.title("Cost function = Mean Cubic Error")
g.show()

h=plt.figure()
plt.plot(alpha,rmse_ame,color="blue")
plt.ylabel("root mean square error")
plt.xlabel("learning rate")
plt.title("Cost function = Absolute Mean Error")
h.show()