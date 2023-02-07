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
library(xgboost)


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
cm
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

#XGBOOST
train1.matrix<-data.matrix(train1)


  for (i in 1:nrow(train1.matrix))
    {
  if (train1.matrix[i,8] ==2 ) train1.matrix[i,8] = 1
  else train1.matrix[i,8] = 0}

train_data <- train1.matrix[,-8]
train_lables <- train1.matrix[,8]

test1.matrix<-data.matrix(test1)
for (i in 1:nrow(test1.matrix))
{
  if (test1.matrix[i,8] ==2 ) test1.matrix[i,8] = 1
  else test1.matrix[i,8] = 0}

test_data <- test1.matrix[,-8]
test_lables <- test1.matrix[,8]


dtrain<- xgb.DMatrix(data =train_data, label=train_lables )
dtest<- xgb.DMatrix(data =test_data, label=test_lables )


fit.xgb <- xgboost(data = dtrain, nrounds=20, objective = "binary:logistic")

pred <- predict(fit.xgb, dtest)
xgpredictclass <- as.numeric(pred>0.5)

cm <- table(xgpredictclass, test_lables)
cm
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
(tp + tn)/(tp + tn + fp + fn)

#ensemble methods - stacking:
train2 <-train1
train2$logit <- predict(logit.reg, train1[,-8], type = "response")
train2$logit<-as.integer(ifelse(train2$logit > 0.5, 1, 0))

train2$rf <- as.integer(predict(x,train1[,-8]))

train2$nb <- predict(fit.nb, train1, type = "class")

train2$xg <- predict(fit.xgb, dtrain)
train2$xg<-as.integer(ifelse(train2$xg > 0.5, 1, 0))

test2 <- test1
test2$logit <- predict(logit.reg, test1[,-8], type = "response")
test2$logit<-ifelse(test2$logit > 0.5, 1, 0)

test2$rf <- predict(x,test1[,-8])

test2$nb <- predict(fit.nb, test1, type = "class")

test2$xg <- predict(fit.xgb, dtest)
test2$xg<-ifelse(test2$xg > 0.5, 1, 0)




#XG boost on ensemble methods
train2.matrix<-data.matrix(train2)


for (i in 1:nrow(train2.matrix))
{
  if (train2.matrix[i,8] ==1 ) train2.matrix[i,8] = 1
  else train2.matrix[i,8] = 0}

for (i in 1:nrow(train2.matrix))
{
  if (train2.matrix[i,10] ==1 ) train2.matrix[i,10] = 1
  else train2.matrix[i,10] = 0}

for (i in 1:nrow(train2.matrix))
{
  if (train2.matrix[i,11] ==1 ) train2.matrix[i,11] = 1
  else train2.matrix[i,11] = 0}


train_data <- train2.matrix[,9:11]
train_lables <- train2.matrix[,8]

test2.matrix<-data.matrix(test2)

for (i in 1:nrow(test2.matrix))
{
  if (test2.matrix[i,8] ==1 ) test2.matrix[i,8] = 1
  else test2.matrix[i,8] = 0}

for (i in 1:nrow(test2.matrix))
{
  if (test2.matrix[i,10] ==1 ) test2.matrix[i,10] = 1
  else test2.matrix[i,10] = 0}

for (i in 1:nrow(test2.matrix))
{
  if (test2.matrix[i,11] ==1 ) test2.matrix[i,11] = 1
  else test2.matrix[i,11] = 0}

test_data <- test2.matrix[,9:11]
test_lables <- test2.matrix[,8]


dtrain<- xgb.DMatrix(data =train_data, label=train_lables )
dtest<- xgb.DMatrix(data =test_data, label=test_lables )


fit.xgb <- xgboost(data = dtrain, nrounds=20, objective = "binary:logistic")

pred <- predict(fit.xgb, dtest)
xgpredictclass <- as.numeric(pred>0.5)

cm <- table(xgpredictclass, test_lables)
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

#adding column US/non US(US=1; non us=0)
for (i in 1:nrow(train.lv2)){
  if (train.lv2$country_destination[i] == "US" ) train.lv2$Destination_category[i] = "1"
  else train.lv2$Destination_category[i] = "0"}


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
actual
predict <- logitPredictClass
cm <- table(predict, actual)
cm
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




#XGBOOST
train1.matrix<-data.matrix(train1)


for (i in 1:nrow(train1.matrix))
{
  if (train1.matrix[i,8] ==2 ) train1.matrix[i,8] = 1
  else train1.matrix[i,8] = 0}

train_data <- train1.matrix[,-8]
train_lables <- train1.matrix[,8]

test1.matrix<-data.matrix(test1)
for (i in 1:nrow(test1.matrix))
{
  if (test1.matrix[i,8] ==2 ) test1.matrix[i,8] = 1
  else test1.matrix[i,8] = 0}

test_data <- test1.matrix[,-8]
test_lables <- test1.matrix[,8]


dtrain<- xgb.DMatrix(data =train_data, label=train_lables )
dtest<- xgb.DMatrix(data =test_data, label=test_lables )


fit.xgb <- xgboost(data = dtrain, nrounds=20, objective = "binary:logistic")

pred <- predict(fit.xgb, dtest)
xgpredictclass <- as.numeric(pred>0.5)

cm <- table(xgpredictclass, test_lables)
cm
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
(tp + tn)/(tp + tn + fp + fn)

#ensemble methods - stacking:
train2 <-train1
train2$logit <- predict(logit.reg, train2[,-8], type = "response")
train2$logit<-ifelse(train2$logit > 0.5, 1, 0)

train2$rf <- predict(x,train1[,-8])

train2$nb <- predict(fit.nb, train1, type = "class")

#train2$xg <- predict(fit.xgb, dtrain)
#train2$xg<-ifelse(train2$xg > 0.5, 1, 0)

test2 <- test1
test2$logit <- predict(logit.reg, test1[,-8], type = "response")
test2$logit<-ifelse(test2$logit > 0.5, 1, 0)

test2$rf <- predict(x,test1[,-8])

test2$nb <- predict(fit.nb, test1, type = "class")

#test2$xg <- predict(fit.xgb, dtest)
#test2$xg<-ifelse(test2$xg > 0.5, 1, 0)


#XG boost on ensemble methods
train2.matrix<-data.matrix(train2)


for (i in 1:nrow(train2.matrix))
{
  if (train2.matrix[i,8] ==1 ) train2.matrix[i,8] = 1
  else train2.matrix[i,8] = 0}

for (i in 1:nrow(train2.matrix))
{
  if (train2.matrix[i,10] ==1 ) train2.matrix[i,10] = 1
  else train2.matrix[i,10] = 0}

for (i in 1:nrow(train2.matrix))
{
  if (train2.matrix[i,11] ==1 ) train2.matrix[i,11] = 1
  else train2.matrix[i,11] = 0}


train_data <- train2.matrix[,9:11]
train_lables <- train2.matrix[,8]

test2.matrix<-data.matrix(test2)

for (i in 1:nrow(test2.matrix))
{
  if (test2.matrix[i,8] ==1 ) test2.matrix[i,8] = 1
  else test2.matrix[i,8] = 0}

for (i in 1:nrow(test2.matrix))
{
  if (test2.matrix[i,10] ==1 ) test2.matrix[i,10] = 1
  else test2.matrix[i,10] = 0}

for (i in 1:nrow(test2.matrix))
{
  if (test2.matrix[i,11] ==1 ) test2.matrix[i,11] = 1
  else test2.matrix[i,11] = 0}

test_data <- test2.matrix[,9:11]
test_lables <- test2.matrix[,8]


dtrain<- xgb.DMatrix(data =train_data, label=train_lables )
dtest<- xgb.DMatrix(data =test_data, label=test_lables )


fit.xgb <- xgboost(data = dtrain, nrounds=20, objective = "binary:logistic")

pred <- predict(fit.xgb, dtest)
xgpredictclass <- as.numeric(pred>0.5)

cm <- table(xgpredictclass, test_lables)
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
#accuracy
(2+9+1++2+1250)/nrow(test1)

