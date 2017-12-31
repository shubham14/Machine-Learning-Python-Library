""" 
This file is the primary file.  This does the following
- Prep the data  in a format suitable for running the image files.
- Create the CNN model 
- Compile the model (tuning is done here).
- Train the model
- FIt the model 
- Get performance metrics.
"""
import numpy as np
import pandas as pd
import glob
import os

import scipy.misc
import matplotlib
import matplotlib.pyplot as plt


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot


from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

category = ["Environmental", "Medical", "Card Game", "Civilization",
            "Economic", "Modern Warfare", "Political", "Wargame",
            "Fantasy", "Territory Building", "Adventure", "Exploration",
            "Fighting", "Miniatures", "Dice", "Movies / TV / Radio theme",
            "Science Fiction", "Industry / Manufacturing", "Ancient", "City Building",
            "Animals", "Farming", "Medieval", "Novel-based",
            "Mythology", "American West", "Horror", "Murder/Mystery",
            "Puzzle", "Video Game Theme", "Space Exploration", "Collectible Components",
            "Bluffing", "Transportation", "Religious", "Travel",
            "Nautical", "Deduction", "Party Game", "Spies/Secret Agents",
            "Word Game", "Mature / Adult", "Renaissance", "Zombies",
            "Negotiation", "Abstract Strategy", "Prehistoric", "Arabian",
            "Aviation / Flight", "Post-Napoleonic", "Trains", "Action / Dexterity",
            "World War I", "World War II", "Comic Book / Strip", "Racing",
            "Real-time", "Humor", "Electronic", "Book",
            "Civil War", "Expansion for Base-game", "Sports", "Pirates",
            "Age of Reason", "none-category", "American Indian Wars", "American Revolutionary War",
            "Educational", "Memory-category", "Maze", "Napoleonic",
            "Print & Play", "American Civil War", "Children's Game", "Vietnam War",
            "Pike and Shot", "Mafia", "Trivia", "Number",
            "Game System", "Korean War", "Music", "Math"]


## This reduces the size of the image down and scales the pixel values
## down and convert it to float.  Since each color depth of the input
## image is a uint8, we scale it by that here.
def preprocess(img, size=(150, 101)):
    """Resize the image size here."""
    img = scipy.misc.imresize(img, size)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img


# Cover art path : All the images retrieved from the internet reside
# in this folder.  Make sure these are the only files in this folder.
path = 'images'
data_file_s = "Data_Modified.csv"
# Get all the image file names using the glob module.
img = glob.glob(os.path.join(path,"*"))


# Read all the images and store them here.  The filename of the image
# and the games are linked via the game_id
img_dict = {}
for i in img:
    try:
        img_val = scipy.misc.imread(i)
        img_dict[int(os.path.basename(i))] = img_val
    except:
        pass
    
# The data file.  We use this to retrieve the data required here.
data = pd.read_csv(data_file_s,sep=',')



def prepare_data(data, img_dict, size=(150, 101)):
    """This is the function that readies all the data required for the
    CNN.
    """
    print("Generation dataset...")



    # Container for the training data.    
    dataset = []
    # Container for the ground truth of labels.
    y = []
    # Container for the labels
    label = []


    # Convenient counter
    cnt = 0

    
    for k,v in img_dict.items():
        cnt = cnt+1         # Iteration counter.
        try:
            if (cnt % 100) == 0: # Scream, "I'm still alive!"
                print "At iteration number", cnt

            # Get game data corresponding to the image 
            data_row = data[data['game_id'] == k]  # x
            
            # 72 is where the categories start and we have 84 of them            
            data_row_trans = data_row.ix[:,72:84+72].as_matrix().T        
            img = preprocess(v,size)
            # If the preprocess failed, don't die.  imread doesn't
            # throw an exception
            if img.shape !=  size+(3,):
                continue

            # Populate the data set 
            dataset.append(img)
            # Populate the labels.
            y.append(data_row_trans)
            # Put all the labels here.
            label.append(k)
        except:
            # In case of corrupted images, just continue.  No need to
            # die.
            continue

        
    return dataset, y, label


    
x,y,l = prepare_data(data,img_dict)

# Split the data 3500 sets and rest.
n_train = 3500
x_train, x_test = x[:3500], x[3500:]
y_train, y_test = y[:3500], y[3500:]
l_train,l_test = l[:3500], l[3500:]


SIZE = (150, 101)


# Create the CNN model using Keras.  Backend is TensorFlow
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(SIZE[0], SIZE[1], 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84, activation='sigmoid')) # Output layer is sigmoid to spit out probabilities


# Compile the model 
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])


# Train the model.
model.fit(np.array(x_train), np.array(y_train).reshape(n_train,84), batch_size=32, epochs=20, verbose=1,
          validation_split=0.2)


# These are the arrays required.
y_test_arr = np.array(y_test).reshape(1303,84)
predictions = model.predict(np.array(x_test)) 

pred_arr = np.copy(predictions)
# Threshold was found manually for the scores.  For what we want
# though, we may actually do with an higher threshold as we are only
# interested in the top 3 classes.
# thresh = 0.2115
thresh = 0.1885
# thresh = 0.1790
# Create filters for the threshold
high_threshold_indices = pred_arr > thresh
low_threshold_indices = pred_arr < thresh
# Threshold the array
pred_arr[high_threshold_indices] = 1
pred_arr[low_threshold_indices] = 0
# Get the F1 scores
print "F1 scores (Macro) ",    f1_score(pred_arr,y_test_arr,average='macro')
print "F1 scores (Micro) ",    f1_score(pred_arr,y_test_arr,average='micro')
print "F1 scores (Samples) ",  f1_score(pred_arr,y_test_arr,average='samples')
print "F1 scores (Weighted)",  f1_score(pred_arr,y_test_arr,average='weighted')


ind_f1_scores =  f1_score(pred_arr,y_test_arr,average=None)
ind_f1_scores_arg =  np.argsort(-f1_score(pred_arr,y_test_arr,average=None))


# We see minor variations in the F1 scores.  This is due to random
# intialization used for weights in the model and multiple minima in
# the optimization space.  We can augment this by running it for a lot
# more epochs or getting better data.
for i,v in enumerate(ind_f1_scores_arg):
    if ind_f1_scores[v]  < 0.1:
        break
    print "F1 score of", category[v], " = ", ind_f1_scores[v]


model.save("model.h5")
