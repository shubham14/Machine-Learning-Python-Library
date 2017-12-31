# ---------- Python program for building a Classifier Chain Model for Multilabel Classification

# Required Libraries
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import hamming_loss,label_ranking_loss,average_precision_score,f1_score,accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import f1_score,precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier

# Reading the data
data=pd.read_csv("Data_Modified.csv",sep=',')

# Segregating the data according to the input features, mechanics and the 
# labels which are to be considered for classification
input_col = ["rank","names","min_players","max_players","avg_time","min_time","max_time","avg_rating","geek_rating","age","category","weight"]

mechanics = ["Action Point Allowance System","Co-operative Play","Hand Management","Point to Point Movement","Set Collection","Trading","Variable Player Powers","Auction/Bidding",
			"Card Drafting","Area Control / Area Influence","Campaign / Battle Card Driven","Dice Rolling","Simultaneous Action Selection","Route/Network Building","Variable Phase Order",
			"Action / Movement Programming","Grid Movement","Modular Board","Storytelling","Area Movement","Tile Placement","Worker Placement","Deck / Pool Building",
			"Role Playing","Memory-mechanic","Partnerships","Pick-up and Deliver","Player Elimination","Secret Unit Deployment","Pattern Recognition","Press Your Luck","Time Track",
			"Voting","Area-Impulse","Hex-and-Counter","Area Enclosure","Pattern Building","Take That","Stock Holding","Commodity Speculation","Simulation","Betting/Wagering","Trick-taking","Line Drawing",
			"Rock-Paper-Scissors","Roll / Spin and Move","Paper-and-Pencil","Acting","Singing","none-mechanic","Chit-Pull System","Crayon Rail System"]

category = ["Environmental","Medical","Card Game","Civilization","Economic","Modern Warfare","Political","Wargame","Fantasy","Territory Building","Adventure","Exploration","Fighting",
			"Miniatures","Dice","Movies / TV / Radio theme","Science Fiction","Industry / Manufacturing","Ancient","City Building","Animals","Farming","Medieval","Novel-based","Mythology",
			"American West","Horror","Murder/Mystery","Puzzle","Video Game Theme","Space Exploration","Collectible Components","Bluffing","Transportation","Religious","Travel","Nautical",
			"Deduction","Party Game","Spies/Secret Agents","Word Game","Mature / Adult","Renaissance","Zombies","Negotiation","Abstract Strategy","Prehistoric","Arabian","Aviation / Flight",
			"Post-Napoleonic","Trains","Action / Dexterity","World War I","World War II","Comic Book / Strip","Racing","Real-time","Humor","Electronic","Book","Civil War","Expansion for Base-game",
			"Sports","Pirates","Age of Reason","none-category","American Indian Wars","American Revolutionary War","Educational","Memory-category","Maze","Napoleonic","Print & Play","American Civil War",
			"Children's Game","Vietnam War","Pike and Shot","Mafia","Trivia","Number","Game System","Korean War","Music","Math"]

# The feature vectors x combined with and the 
# associated labels y
x = data[input_col+mechanics]
y = data[category]

#label encoding 
d = x.apply(LabelEncoder().fit_transform)
d = d.as_matrix()
#The results might vary due to the usage of random state with train and test split
X_train, X_test, y_train, y_test = train_test_split(d,y,test_size=0.2, random_state=42)
 
# The classifier instance with the classifier as 
# RandomForestClassifier
clf_cc = ClassifierChain(RandomForestClassifier(n_estimators=100,max_depth=200))

#fitting the model for the classification into the labels
clf_cc.fit(X_train,y_train.astype(float))
#predictions
predictions_cc = clf_cc.predict(X_test)
pred_prob = clf_cc.predict_proba(X_test)

#Finding the evaluation metrics 
# micro recall, macro recall, micro precision, macro precision
# micro f1, macro f1, hamming loss
r1 = recall_score(y_true=y_test, y_pred=predictions_cc, average='micro')
r2 = recall_score(y_true=y_test, y_pred=predictions_cc, average='macro')
p1 = precision_score(y_true=y_test, y_pred=predictions_cc, average='micro')
p2 = precision_score(y_true=y_test, y_pred=predictions_cc, average='macro')
f1 = f1_score(y_true=y_test, y_pred=predictions_cc, average='micro')
f2 = f1_score(y_true=y_test, y_pred=predictions_cc, average='macro')
Score_cc_ham = hamming_loss(y_test,predictions_cc)

# Printing the evaluation metrics
print "Hamming Loss for classifier chains", Score_cc_ham
print "The micro recall is", r1
print "The macro recall is", r2
print "The micro precision is", p1
print "The macro precision is", p2
print "The micro F-1 score is", f1
print "The macro F-1 score is", f2