import math
import csv
import pandas as pd
import os

def LinearRegession(x,y):
	SumX=sum(x)
	SumY=sum(y)
	SSumXX=0
	SSumXY=0
	for i in range(0,len(x)):
		SSumXX=x[i]*x[i]
		SSumXY=x[i]*y[i]
	n=len(x)
	a=(n*SSumXY-SumX*SumY)/(n*SSumXX-SumX*SumX)
	b=(SSumXX*SumY-SumX*SSumXY)/(SSumXX*n-SumX*SumX)
	return a,b

def predict(x,y,xObs):
	[a,b]=LinearRegession(x,y)
	return a*xObs+b
