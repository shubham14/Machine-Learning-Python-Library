''' Python functions to implement Logistic Regression'''

import math
import csv
import pandas as pd
import numpy as np

#Activation function is choosen as sigmoid
def sigmoid(x):
	y=1+math.exp(-x)
	return 1/y

#Computation of the cost function
def Cost(t,x,y):
	m=X.shape[0]
	t=reshape(t,len(t),1)
	J = (1./m) * (-transpose(y).dot(log(sigmoid(X.dot(t)))) - transpose(1-y).dot(log(1-sigmoid(X.dot(t)))))
    grad = transpose((1./m)*transpose(sigmoid(X.dot(t)) - y).dot(X))
    return J[0][0]

#Compute Gradient
def Grad(t,x,y):
	h=sigmoid(x.dot(t.T))
	d=h-y
	grad=zeros(t.shape[0])
	l=grad.size
	for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / m) * sumdelta * - 1
    theta.shape = (3,)
    return grad

#Making Predictions
def Predict(t,X):
	m,n=X.shape
	p=zeros(m)
	h=sigmoid(X.dot(t.T))
	for i in range(0, h.shape[0]):
        if h[i] > 0.5:
            p[it, 0] = 1
        else:
            p[i, 0] = 0
    return p
