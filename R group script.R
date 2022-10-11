rm(list = ls())
cat('\014')
setwd("~/Documents/1.MSBA/Spring_22/2. R(BUAN 6356)/group project/airbnb-recruiting-new-user-bookings")
library('dplyr')
library("vtree")
library("parallel")
library(caret)
library(caTools)
library(e1071)
library(rpart.plot)
library(rpart)
library(randomForest)
library(ROCR)
library(tidyverse)


data = read.csv('train_users_2.csv')


#Feature selection:
data1 <- subset(data, select =c(signup_method, country_destination, signup_app, affiliate_provider, affiliate_channel, language, gender, age))

#Cleaning gender column:
y <- which(data1$gender == "-unknown-")
data2 <- data1[-y,]
data2 = na.omit(data2)

#feature engineering - age:
for (i in 1:nrow(data2)){
  if (data2$age[i] < 21) data2$age[i] = "yonger than 20"
  if (data2$age[i] >= 21 & data2$age[i] < 31) data2$age[i] = "20-30"
  if (data2$age[i] >= 31 & data2$age[i] < 41) data2$age[i] = "30-40"
  if (data2$age[i] >= 41 & data2$age[i] < 51) data2$age[i] = "40-50"
  if (data2$age[i] >= 51 & data2$age[i] < 61) data2$age[i] = "50-60"
  if (data2$age[i] >= 60) data2$age[i] = "older than 60"
}

train <- data2

#Adding New feature - Booked(whether customer booked or not boked)::
for (i in 1:nrow(train)){
if (train$country_destination[i] =="NDF" ) train$Book[i] = "No-booking"
else train$Book[i] = "Booked"}

table(train$Book)


#converting to factors:
train$signup_method <- as.factor(train$signup_method)
train$signup_app <- as.factor(train$signup_app)
train$affiliate_channel <- as.factor(train$affiliate_channel)
train$affiliate_provider <- as.factor(train$affiliate_provider)
train$language <- as.factor(train$language)
train$gender <- as.factor(train$gender)
train$age <- as.factor(train$age)
train$country_destination <- as.factor(train$country_destination)
train$Book <- as.factor(train$Book)


levels(train$signup_method)
levels(train$signup_app)
levels(train$affiliate_provider)
levels(train$affiliate_channel)


plot(train$country_destination, main = "Distribution of output classes", xlab ="Countries", ylab ="Count")


#LEVEL-1(Book vs no book)
#Logistic regression:
#sampling
set.seed(2)

train.lv1 <- subset(train, select=-c(country_destination))

sample <- sample(1:nrow(train.lv1), (0.8)*nrow(train.lv1))
train1 <- train.lv1[sample,]
test1 <- train.lv1[-sample,]

plot(train.lv1$Book, main = "Distribution of Book vs no Book", xlab ="Classes", ylab ="Count")



#model
logit.reg <- glm(Book ~. , 
                 data = train1, family = "binomial") 
#predict
logitPredict <- predict(logit.reg, test1[,-8], type = "response")
logitPredict
# we choose 0.5 as the cutoff here for 1 vs. 0 classes
logitPredictClass <- ifelse(logitPredict > 0.5, 1, 0)



actual <- test1$Book
predict <- logitPredictClass
cm <- table(predict, actual)
# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
(tp + tn)/(tp + tn + fp + fn)




#random forest
x<-randomForest(factor(Book)~.,data=train1, importance=TRUE, ntree=500)

importance(x)
varImpPlot(x)
Prediction<-predict(x,test1[,-8])
table(actual=test1[,8],Prediction)




wrong<-(test1[,8]!=Prediction)
error_rate<-sum(wrong)/length(wrong)
error_rate
accuracy <- (1-error_rate)*100
accuracy



#Naive bayes


library(e1071)

fit.nb <- naiveBayes(Book ~ ., data = train1)
fit.nb



# Evaluate Performance using Confusion Matrix
actual <- test1$Book
# predict class probability
nbPredict <- predict(fit.nb, test1, type = "raw")
# predict class membership
nbPredictClass <- predict(fit.nb, test1, type = "class")
cm <- table(nbPredictClass, actual)
cm
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
(tp + tn)/(tp + tn + fp + fn)


#LEVEL-2(US vs Non US)
#Removing Book column
train.lv2 <- subset(train, select=-c(Book))
train.lv2$country_destination <- as.character(train.lv2$country_destination)



#removing NDF(no booking)
y <- which(train.lv2$country_destination == "NDF")
train.lv2 <- train.lv2[-y,]


table(train.lv2$country_destination)
train.lv2$country_destination <- as.factor(train.lv2$country_destination)

#adding column US/non US
for (i in 1:nrow(train.lv2)){
  if (train.lv2$country_destination[i] == "US" ) train.lv2$Destination_category[i] = "US"
  else train.lv2$Destination_category[i] = "Non-Us"}

train.lv2 <- subset(train.lv2, select=-c(country_destination))
train.lv2$Destination_category <- as.factor(train.lv2$Destination_category)

plot(train.lv2$Destination_category, main ="Distribution of US vs Non-US", xlab ="Classes", ylab ="Count" )

#Models
#Logistic regression:
sample <- sample(1:nrow(train.lv2), (0.8)*nrow(train.lv2))
train1 <- train.lv2[sample,]
test1 <- train.lv2[-sample,]


logit.reg <- glm(Destination_category ~. , 
                 data = train1, family = "binomial") 
#predict
logitPredict <- predict(logit.reg, test1[,-8], type = "response")
logitPredict
# we choose 0.5 as the cutoff here for 1 vs. 0 classes
logitPredictClass <- ifelse(logitPredict > 0.5, 1, 0)



actual <- test1$Destination_category
predict <- logitPredictClass
cm <- table(predict, actual)
# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy 71%
(tp + tn)/(tp + tn + fp + fn)




#random forest
x<-randomForest(factor(Destination_category)~.,data=train1, importance=TRUE, ntree=500)

importance(x)
varImpPlot(x)
Prediction<-predict(x,test1[,-8])
table(actual=test1[,8],Prediction)




wrong<-(test1[,8]!=Prediction)
error_rate<-sum(wrong)/length(wrong)
error_rate
accuracy <- (1-error_rate)*100
accuracy



#Naive bayes
library(e1071)

fit.nb <- naiveBayes(Destination_category ~ ., data = train1)
fit.nb



# Evaluate Performance using Confusion Matrix
actual <- test1$Destination_category
# predict class probability
nbPredict <- predict(fit.nb, test1, type = "raw")
# predict class membership
nbPredictClass <- predict(fit.nb, test1, type = "class")
cm <- table(nbPredictClass, actual)
cm
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
(tp + tn)/(tp + tn + fp + fn)




#LEVEL3(Which country?)
#removing Book column
train.lv3 <- subset(train, select=-c(Book))
train.lv3$country_destination <- as.character(train.lv3$country_destination)

#removing NDF(no booking cases)
y <- which(train.lv3$country_destination == "NDF")
train.lv3 <- train.lv3[-y,]
table(train.lv3$country_destination)

#removing US(no US cases)
y <- which(train.lv3$country_destination == "US")
train.lv3 <- train.lv3[-y,]
table(train.lv3$country_destination)
train.lv3$country_destination <- as.factor(train.lv3$country_destination)

#Sampling:
sample <- sample(1:nrow(train.lv3), (0.8)*nrow(train.lv3))
train1 <- train.lv3[sample,]
test1 <- train.lv3[-sample,]


plot(train.lv3$country_destination, main ="Distribution of Non-US countries", xlab ="Classes", ylab ="Count" )


#Multinomial model

library(nnet)
model <- nnet::multinom(country_destination ~., data = train1)

#predict
logitPredict <-model %>% predict(test1)
logitPredict


actual <- test1$country_destination
predict <- logitPredict
cm <- table(predict, actual)
cm
#accuracy - 38%
(2+9+1++2+1250)/nrow(test1)

