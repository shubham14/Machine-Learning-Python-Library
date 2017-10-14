from __future__ import division
from collections import defaultdict
import math
import re
import os
import csv
import pandas as pd
import numpy as np
import random
from scipy.misc import logsumexp

train_file=open('SPARSE.TRAIN.50','r+')
Vocab=open('TOKENS_LIST','r+')
test_file=open('SPARSE.TEST','r+')

def createDict(file1):
	data=[]
	for line in file1:
		data.append(line)
	r=[]
	doc_list=[]
	labels=[]
	for ele in data:
		labels.append(int(ele.split('  ')[0]))

	for i in range(len(data)):
		VocabList=[]
		VocabCount=[]
		start=data[i].find('  ')+2
		end=data[i].find('\n',start)
		r=re.findall(r'\w+:\w+',data[i][start:end])
		for ele in r:
		    VocabList.append(int(ele.split(':')[0]))
		    VocabCount.append(int(ele.split(':')[1]))
		#doc_Vocab_list.append(Vocab_list)
		#doc_Vocab_count.append(Vocab_count)
		'''list of length 2144 having dictionary elements with Vocabulary and count'''
		Dict_1=dict(zip(VocabList,VocabCount))
		doc_list.append(Dict_1)
	return labels,doc_list

#training data parsed
[labels,doc_list]=createDict(train_file)

def calc_priorProb(class_label,labels):
	pos_count=labels.count(1)
	neg_count=labels.count(-1)

	P_pos_class=pos_count/float(pos_count+neg_count)
	P_neg_class=neg_count/float(pos_count+neg_count)
	if class_label==1:
		return P_pos_class
	else:
		return P_neg_class

#Defining word probabilities and word counts
Word_count_class_pos=defaultdict(int)
#Word_prob_class_pos=dict()
Word_count_class_neg=defaultdict(int)
#Word_prob_class_neg=dict()
Word_prob_class_neg=[]
Word_prob_class_pos=[]

for i in range(1448):
	count_word_p=0
	count_word_n=0
	for j in range(len(doc_list)):
		if labels[j]==1:
			if doc_list[j].has_key(i+1):
				count_word_p+=doc_list[j][i+1]
		if labels[j]==-1:
			if doc_list[j].has_key(i+1):
				count_word_n+=doc_list[j][i+1]
	
	Word_count_class_neg[i+1]=count_word_n
	Word_count_class_pos[i+1]=count_word_p


#calculating word counts
# for i in range(1448):
# 	s_neg+=(1+Word_count_class_neg[i+1])
s_pos=sum(Word_count_class_pos.values())+1406
s_neg=sum(Word_count_class_neg.values())+1424

for i in range(1448):
	Word_prob_class_pos.append(float(1+int(Word_count_class_pos[i+1]))/float(s_pos))

for i in range(1448):
	Word_prob_class_neg.append(float(1+int(Word_count_class_neg[i+1]))/float(s_neg))

#prediction using test data
[labels_test,doc_list_test]=createDict(test_file)
labels_pred=[]
doc_prod_prob=[]
for i in range(len(doc_list_test)):
	#keys=[]
	p1=0
	p2=0
	doc_words=doc_list_test[i].keys()
	for ele in doc_words:
		p1+=doc_list_test[i][ele]*np.log(Word_prob_class_pos[ele-1])
		p2+=doc_list_test[i][ele]*np.log(Word_prob_class_neg[ele-1])
	doc_prod_prob.append([p1,p2])
	#ans=float(doc_prod_prob[i][0]*calc_priorProb(1,labels))/(float(doc_prod_prob[i][0]*calc_priorProb(1,labels))+float(doc_prod_prob[i][1]*calc_priorProb(-1,labels)))
	
for j in range(len(doc_prod_prob)):
	if doc_prod_prob[j][0]==0 and doc_prod_prob[j][1]==0:
		labels_pred.append(1)
	else:
		a=doc_prod_prob[j][0]+np.log(calc_priorProb(1,labels))
		b=doc_prod_prob[j][1]+np.log(calc_priorProb(-1,labels))
		denm=np.logaddexp(a,b)
		ans=np.exp(float(doc_prod_prob[j][0])-denm)
		if ans>=0.5:
			labels_pred.append(1)
		else:
			labels_pred.append(-1)

#print (labels_pred),len(labels_test)
#Calculate error
s=0
for i in range(len(labels_test)):
	if labels_pred[i]!=labels_test[i]:
		s+=1

error=float(s)/float(len(labels_test))

#Words which are the most indicative of the document being spam or not:
#build Vocab map
Vocab_map=dict()
vocab_list=[]
for line in Vocab:
	vocab_list.append(line)

for ele in vocab_list:
	Vocab_map[int(ele.split(' ')[0])]=ele.split(' ')[1][:-1]

#Calculating log probabilities
log_measure=[]
for i in range(len(vocab_list)):
	log_measure.append(np.log(float(Word_prob_class_pos[i])/float(Word_prob_class_neg[i])))

log_sort_1 = np.argsort(log_measure)[::-1][:5]
Vocab_map[615]
print log_sort_1
Spam_indicative_words=[]
# for i in range(5):
for ele in log_sort_1:
	Spam_indicative_words.append(vocab_list[ele][:-1])
# 	Spam_indicative_words.append(Vocab_map[log_sort_1(i)])

print Spam_indicative_words,error