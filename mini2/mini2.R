library(mlbench)

########################################
data("Sonar")
data<-Sonar
data<-data[complete.cases(data), ]
# data<-data[,-1]
classind <- 61
names(data)[classind]<-"class"

########################################
data("Glass")
data<-Glass
data<-data[complete.cases(data), ]
# data<-data[,-1]
classind <- 10
names(data)[classind]<-"class"

##########################################
# method_names = c("CART","RF","knn","lda","qda","pda","nb","mda")
# methods = c("rpart","ranger","knn","lda","qda","pda","nb","mda")
method_names = c("CART","RF","knn","lda","pda","nb","mda")
methods = c("rpart","ranger","knn","lda","pda","nb","mda")

B<-5
iERRMAT<-matrix(0,B,7)

ctrl<-trainControl(method="cv",summaryFunction=multiClassSummary)

for (b in (1:B)) {
  inTrain<-createDataPartition(data$class,p=3/4,list=FALSE)
  trainW<-data[inTrain,-classind]
  testW<-data[-inTrain,-classind]
  trainwY<-data$class[inTrain]
  testwY<-data$class[-inTrain]
  i<-1
  for (method in methods) {
    print(method)
    fit<-caret::train(class~.,data=data[inTrain,],method=method,tuneLength=15,trControl=ctrl)
    pp<-predict(fit,newdata=data[-inTrain,-classind],type="raw")
    iERRMAT[b,i]<-length(pp[pp!=data$class[-inTrain]])/length(pp)
    i<-i+1
  }
  print(b)
}
par(mfrow=c(1,1))
boxplot(iERRMAT,names=method_names)

##################################
bagging_method_names = c("CART","RF","knn","lda","mda", "stacked")
bagging_methods = c("classif.rpart","classif.ranger","classif.knn","classif.lda","classif.mda")
B<-5
bagERRMAT<-matrix(0,B,length(bagging_method_names))
for (b in (1:B)) {
  inTrain<-createDataPartition(data$class,p=3/4,list=FALSE)
  trainData<-data[inTrain,]
  testData<-data[-inTrain,]
  tskTrain = makeClassifTask(data=trainData,target="class")
  tskTest = makeClassifTask(data=testData,target="class")
  i<-1
  stacking_learners = list()
  for (method in bagging_methods) {
    print(method)
    learner = makeBaggingWrapper(method)
    stacking_learners[[i]] = learner
    mdl = train(learner,tskTrain)
    prd = predict(mdl,tskTest)
    predictions = prd$data$response
    bagERRMAT[b,i]<-length(predictions[predictions!=testData$class])/length(predictions)
    print(bagERRMAT[b,i])
    i<-i+1
  }
  print("stacked")
  stack_learner = makeStackedLearner(stacking_learners, method="average", predict.type = "prob")
  mdl = train(stack_learner,tskTrain)
  prd = predict(mdl,tskTest)
  predictions = prd$data$response
  bagERRMAT[b,i]<-length(predictions[predictions!=testData$class])/length(predictions)
  print(bagERRMAT[b,i])
  print(b)
}
par(mfrow=c(1,1))
boxplot(bagERRMAT,names=bagging_method_names)
