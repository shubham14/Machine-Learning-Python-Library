""" This is a small script to get all the images from the image url """
import urllib
import os
import pandas as pd
import os 


data = pd.read_csv("Data_Modified.csv",sep=',')
cover_art_folder="images/"


# zip(data.image_url, data.game_id, data['rank'][):
for i,d in data.iterrows():    
    if(not os.path.isfile(os.path.join(cover_art_folder,str(d.game_id)))):
        if (not (d.image_url == 'none')):            
            urllib.urlretrieve(d.image_url, os.path.join(path,str(d.game_id)))
