''' Python Functions to implement Naive Bayes method'''

import csv
import pandas as pd
import numpy as np
import random
import os
import math

#load CSV for processing
def loadCsv(name):
	CFile=csv.reader(open(name,"rb"))
	dataset=list(CFile)
	for i in range(len(dataset)):
		dataset[i]=[float(x) for x in dataset[i]]
	return dataset

#Calculates Mean
def mean(l):
	return sum(l)/float(len(l))

#Calculates Standard Deviation
def StdDev(l):
	M=mean(l)
	var=sum([pow(x-M,2)] for e in l)/float(len(l)-1)
	return math.sqrt(var)

#Splits the given Dataset for obtaining the training and test sets
def split(dataset,Ratio):
	trainSize=int(len(dataset)*Ratio)
	TrainSet=[]
	TestSet=list(dataset)
	while len(TrainSet)<trainSize:
		i=random.randint(0,len(TestSet))
		TrainSet.append(TestSet.pop(i))
	return [TrainSet,TestSet]

#Splits the dataset accorsing to class labels
def SplitLabels(dataset):
	D=list(dataset)
	sep={}
	for i in range(len(D)):
		vec=dataset[i]
		if vec[-1] not in sep:
			sep[vec[-1]]=[]
		sep[vec[-1]].append(vec)
	return sep

def Probability(x, mean, SD):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(SD,2))))
	return (1 / (math.sqrt(2*math.pi) * SD)) * exponent

#Class summaries
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries


def ClassProbability(X,summaries):
	prob={}
	for labels,classSum in summaries.iteritems():
		prob[labels]=1
		for i in range(len(classSum)):
			u,SD=classSum[i]
			x=X[i]
			prob[labels]*=Probability(x,u,SD)
	return prob

#Make predictions based on probabilities
def predict(summaries,X):
	prob=ClassProbability(X,summaries)
	PLabel,P=None,-1
	for Label, probability in prob.iteritems():
		if PLabel is None or probability>P:
			PLabel=Label
			P=probability
	return Plabel


def predict(X):
