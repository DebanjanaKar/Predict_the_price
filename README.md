# Predict_the_price
Prediction of house prices based on different features &amp; experimenting with different optimization algorithms &amp; cost functions.


**********************READ ME File****************************************************

@author - Debanjana Kar


*****General********

1.Along with each source code there is a corresponding folder containing it's results.
2.Each folder contains a .csv of final predicted values,a text document containing learnt parameters & RMSE values, & .png of the plots.
3.The report too contains the learnt parameters & RMSE values along with required justifications.
4.To run each source code file, use the .csv file provided along in this folder or convert the data set that has been given in the assignment to a .csv file first.
5.Change the path to required path in the system being used.
6.Run the python file.

************ Linear Regression Implementation - linearRegression.py *********

1.The module grad_Descent_train() trains the model.
2.The module grad_Descent_test() tests the model.
3.If specific values are required to be seen , like error and theta at every iteration, please uncomment the corresponding print statements in the file.


************ Experimenting with Optimization Algorithms - optimization_algo.py ***********************************

1.The module grad_Descent_train() trains the model using gradient descent algorithm.
2.The module irls_train() trains the model using iterative reweighted least square algorithm.
3.The module test() tests the model.
4.If specific values are required to be seen , like error and theta at every iteration, please uncomment the corresponding print statements in the file.
5.Apart from the required RMSE vs. learning rate graphs for each algorithm, the code generates mean square error vs. # epochs for each algorithm as well.


************ Experimenting with combination of features - combination_of_features.py *********************************

1.The module grad_Descent_train() trains the model for each combination of features.
2.The module grad_Descent_test() tests the model for each combination of features.
3.Required RMSE vs. learning rate graphs for each combination is obtained.


*********** Experimenting with cost functions - cost_function.py *****************************************************

1.The module grad_Descent_train_mse() trains the model using mean squared error.
2.The module grad_Descent_train_mce() trains the model using mean cubic error.
3.The module grad_Descent_train_ame() trains the model using absolute mean error.
4.The module grad_Descent_test() tests the model using root mean squared error for the different cost functions.
5.Required RMSE vs. learning rate graphs for each combination is obtained.


______________________________________________________________________________________________________________________
