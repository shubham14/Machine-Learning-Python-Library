# Readme file for executing scripts for multilabel classification of board games

# The link is also available at :

#Datasets
# Link to the dataset
https://bitbucket.org/akaberto/multilabelclassifier/src/0958ee2af86485004e2e1c9e9c2f796fd8033e78/src/?at=master
# Save the dataset as the name "Data_Modified.csv"

# Image Dataset for the CNN
https://umich.box.com/s/9pdgkx7jehnxedx9pc40f1hulk4x2zjk


# Binary Relevance 

	# The code has been commented 

	Required environment : R version 3.3.1

	Dependencies (Libraries) required for executing the script are: 	
	1) utiml - To install, type install.package("utiml")
	2) mldr - To install, type install.package("mldr")

	# Steps to execute the script for using Binary Relevance for multilabel classification:

	a) If IDE like Rstudio is not installed on the system :
		1) On an R terminal, navigate to the directory where the R file is stored, type in:

		RScript MLL_Binary\ Relevance.R

		2) The results for the evaluation metrics would be obtained from the column named 'results2' in the dataframe that is obtained after execution.

	b) If IDE like RStudio is installed on the system:
		1) Navigate to the directory where R file is stored and open it.
		2) Click on the Run button. The results for the evaluation metrics would be obtained from the column named 'results2' in the dataframe that is obtained after execution.

# Classifier Chains:
	
	# The code has been commented

	Required environment: 
	Python 2.7 

	Dependencies (Libraries) required for executing the script are:
	1) Pandas - To install, type pip install pandas
	2) Numpy - To install, type pip install numpy
	3) Skmultilearn (which is a multilabel classification library built on top of sklearn) - To install, type pip install scikit-multilearn
	4) Sklearn -  To install, type pip install sklearn

	# Steps to execute the script for using Classifier Chains for multilabel classification:

		1) On a python terminal access the folder where the python file is stored, type in:

		python Classifer_Chains.py

		2) The results for the evaluation metrics would be displayed in the python terminal


# Convolutional Neural Networks

	# The code has in line comments 
		
	Required environment: 
	Python 2.7 

	Dependencies (Libraries) required for executing the script are:
	1) Pandas - To install, type pip install pandas
	2) Numpy - To install, type pip install numpy
	3) Sklearn -  To install, type pip install sklearn
	4) keras - To install, type pip install keras
	5) tensorflow - To install, type pip install tensorflow
	6) scipy - To install, type pip install scipy
	7 h5py -  To install, type pip install h5py
	
	# To execute, Do the following 
		1) Specify where the image folder and the Data_Modified.csv is present.  Change the path and data_file_s variables in the file.
		2) Enter python rankme.py on a terminal with the above prerequisites.
		3) The results will be displayed in the terminal.
	
    # Note.  You need to run the get_images.py before you run the script.  You have to specify the following
		
		1) Change data variable in the script to point to the path of the csv file used in this project.
		
		2) Specify where you want to place the images 

		3) The results are displayed in the python terminal 

