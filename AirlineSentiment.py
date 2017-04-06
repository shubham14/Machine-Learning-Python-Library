import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

data=pd.read_csv('Tweets.csv')
print (data.head())

count_Airline=data['airline'].value_counts()

def plotAirlineSentiment(airline):
	df=data[data['airline']==airline]
	count=df['airline_sentiment'].value_counts()
	Index=[1,2,3]
	plt.bar(Index,count)
	plt.xticks(Index,['negative','neutral','positive'])
	plt.ylabel('Mood Count')
	plt.xlabel('Mood')
	plt.title('Count of Moods')

def NR_Count(airline):
	if airline=='All' or airline=='all':
		df=data
	else:
		df=data[data['airline']==airline]
	count=dict(df['negativereason'].value_counts())
	Unique_reason=list(data['negativereason'].unique())
	Unique_reason=[x for x in Unique_reason if str(x)!='nan']
	Reason_frame=pd.DataFrame({'Reasons':Unique_reason})
	Reason_frame['count']=Reason_frame['Reasons'].apply(lambda x:count[x])
	return Reason_frame


NR_count=dict(data['negativereason'].value_counts(sort=False))
print NR_count

def tweet_to_words(raw_tweet):
	letters_only=re.sub("[^a-zA-Z]"," ",raw_tweet)
	words=letters_only.lower().split()
	stops=set(stopwords.words("english"))
	useful_words=[w for w in words if w not in stops]
	return (" ".join(useful_words))

def clean_tweet_length(raw_tweet):
	letters_only=re.sub("[^a-zA-Z]"," ",raw_tweet)
	words=letters_only.lower().split()
	stops=set(stopwords.words("english"))
	useful_words=[w for w in words if w not in stops]
	return (len(useful_words))

data['clean_tweet']=data['text'].apply(lambda x: tweet_to_words(x))
data['Tweet_length']=data['text'].apply(lambda x: clean_tweet_length(x))
data['sentiment']=data['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)
train,test = train_test_split(data,test_size=0.2,random_state=42)

train_clean_tweet=[]
for tweet in train['clean_tweet']:
	train_clean_tweet.append(tweet)
test_clean_tweet=[]
for tweet in test['clean_tweet']:
	test_clean_tweet.append(tweet)


'''Feature Extraction'''
v=CountVectorizer(analyzer="word")
train_features=v.fit_transform(train_clean_tweet)
test_features=v.fit_transform(test_clean_tweet)

dense_train_features=train_features.toarray()
dense_test_features=test_features.toarray()
Accuracy=[]
Model=[]
Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]

for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(train_features,train['sentiment'])
        pred = fit.predict(test_features)
    accuracy = accuracy_score(pred,test['sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))    
