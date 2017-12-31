# BINARY RELEVANCE METHOD

# Reading the Data and using the right attributes

data<-read.csv("Data_Modified.csv")
data<-data[,1:156] #removing 2805 designers
data_modified<-data[,-c(1,2,3,4,11,12,14,16,18,19)] #removing unnecessary attributes
data_modified<-data_modified[,-c(7,9)] #removing unnecessary attributes

# Separate into training (80%) and test (20%)

set.seed(10)
sample<-sample(1:4999,1000)

data_modified$train=1
data_modified$train[sample]=0

data_modified=data_modified[,c(145,1:144)] #making training column as first column
sum(colSums(data_modified[,10:145])<2) #check if any mechanics/categories are less than 2 in a column for overidentification

train=subset(data_modified,train==1) #3999
test=subset(data_modified,train==0) #1000 (20% test data)

# Create a multi-label dataset

library(utiml)
library(mldr)

train_attrib1=train[,2:9] #8 main attributes
train_attrib1=scale(train_attrib1) #scaling 8 main attributes
train_sub=cbind(train_attrib1,train[,10:145]) #combining with 52 mechanics and 84 categories

test_attrib1=test[,2:9] #8 main attributes
test_attrib1=scale(test_attrib1) #scaling 8 main attributes
test_sub=cbind(test_attrib1,test[,10:145]) #combining with 52 mechanics and 84 categories

mldr_obj<-mldr_from_dataframe(train_sub,labelIndices = c(seq(61,144)),name="train") #create mldr object
mldr_obj_test<-mldr_from_dataframe(test_sub,labelIndices = c(seq(61,144)),name="test")

# Apply Binary Relevance Method

brmodel<-br(mldr_obj,"RF",cores=1,seed=123,ntree=100) # can use RF, SVM, etc.
#save(brmodel,file="br_model_100.Rdata")
#load("br_model_100.Rdata")
prediction_br<-predict(brmodel,mldr_obj_test)

result1 <- multilabel_evaluate(mldr_obj_test, prediction_br, "bipartition")
result2 <- multilabel_evaluate(mldr_obj_test, prediction_br, "label-based") #reporting label based metrics
result3 <- multilabel_evaluate(mldr_obj_test, prediction_br, "example-based")

result1<-as.data.frame(result1)
result2<-as.data.frame(result2)
colnames(result2)<-"label-based"
result3<-as.data.frame(result3)
colnames(result3)<-"example-based"

