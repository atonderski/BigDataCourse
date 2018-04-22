#### If you want to apply caret to multiclass data, the classes have to have LETTER names like "A", "B", "C" and 
#### you have to change to multiclassSummary and use a different metric like Accuracy instead of ROC.
wine2<-wine
wine2$class[wine$class==1]<-"A"
wine2$class[wine$class==2]<-"B"
wine2$class[wine$class==3]<-"C"


library(scatterplot3d)
par(mfrow = c(2, 2))
mar0 = c(2, 3, 2, 3)
scatterplot3d(wine2[, 2], wine2[, 3], wine[, 4], mar = mar0, color = c("blue",
                                                                       "black", "red")[wine$class], pch = 19)
scatterplot3d(wine2[, 3], wine2[, 4], wine[, 5], mar = mar0, color = c("blue",
                                                                       "black", "red")[wine$class], pch = 19)
scatterplot3d(wine2[, 4], wine2[, 5], wine[, 2], mar = mar0, color = c("blue",
                                                                       "black", "red")[wine$class], pch = 19)
scatterplot3d(wine2[, 5], wine2[, 2], wine[, 3], mar = mar0, color = c("blue",
                                                                       "black", "red")[wine$class], pch = 19)


library(klaR)
mywine<-wine
newfac<-rep(0,dim(wine)[1])
newfac[wine$class==1]<-"Zero"
newfac[wine$class==2]<-"One"
newfac[wine$class==3]<-"Two"
newfac<-as.factor(newfac)
mywine$class<-newfac

### Check classification boundaries between classes
partimat(class~.,data=mywine[,c(1,2:5)],method="lda")

partimat(class~.,data=mywine[,c(1,2:5)],method="qda")

partimat(class~.,data=mywine[,c(1,2:5)],method="naiveBayes")

partimat(class~.,data=mywine[,c(1,2:5)],method="rpart")

#################################################################33

inTrain<-createDataPartition(wine2$class,p=3/4,list=FALSE)
trainW<-wine2[inTrain,-1]
testW<-wine2[-inTrain,-1]
trainwY<-wine2$class[inTrain]
testwY<-wine2$class[-inTrain]


ctrl<-trainControl(method="repeatedcv",repeats=10,summaryFunction=multiClassSummary)
rpartfit<-caret::train(class~.,data=wine2[inTrain,],method="rpart",tuneLength=15,trControl=ctrl)
plot(rpartfit)
knnfit<-caret::train(class~.,data=wine2[inTrain,],method="knn",tuneLength=15,trControl=ctrl)
plot(knnfit)
knnfit$finalModel


pdafit<-caret::train(class~.,data=wine2[inTrain,],method="pda",tuneLength=15,trControl=ctrl)
plot(pdafit)

pp<-predict(rpartfit, newdata=wine2[-inTrain,-1],type="raw")
table(pp,wine2$class[-inTrain])
length(pp[pp!=wine2$class[-inTrain]])/length(pp)

B<-5
ERRMAT<-matrix(0,B,8)

ctrl<-trainControl(method="cv",summaryFunction=multiClassSummary)

for (b in (1:B)) {
  inTrain<-createDataPartition(wine2$class,p=3/4,list=FALSE)
  trainW<-wine2[inTrain,-1]
  testW<-wine2[-inTrain,-1]
  trainwY<-wine2$class[inTrain]
  testwY<-wine2$class[-inTrain]
  fit<-caret::train(class~.,data=wine2[inTrain,],method="rpart",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=wine2[-inTrain,-1],type="raw")
  ERRMAT[b,1]<-length(pp[pp!=wine2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=wine2[inTrain,],method="ranger",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=wine2[-inTrain,-1],type="raw")
  ERRMAT[b,2]<-length(pp[pp!=wine2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=wine2[inTrain,],method="knn",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=wine2[-inTrain,-1],type="raw")
  ERRMAT[b,3]<-length(pp[pp!=wine2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=wine2[inTrain,],method="lda",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=wine2[-inTrain,-1],type="raw")
  ERRMAT[b,4]<-length(pp[pp!=wine2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=wine2[inTrain,],method="qda",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=wine2[-inTrain,-1],type="raw")
  ERRMAT[b,5]<-length(pp[pp!=wine2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=wine2[inTrain,],method="pda",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=wine2[-inTrain,-1],type="raw")
  ERRMAT[b,6]<-length(pp[pp!=wine2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=wine2[inTrain,],method="nb",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=wine2[-inTrain,-1],type="raw")
  ERRMAT[b,7]<-length(pp[pp!=wine2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=wine2[inTrain,],method="mda",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=wine2[-inTrain,-1],type="raw")
  ERRMAT[b,8]<-length(pp[pp!=wine2$class[-inTrain]])/length(pp)
  print(b)
}
par(mfrow=c(1,1))
boxplot(ERRMAT,names=c("CART","RF","knn","lda","qda","pda","nb","mda"))
##########################################
B<-5
iERRMAT<-matrix(0,B,8)

ctrl<-trainControl(method="cv",summaryFunction=multiClassSummary)

iris2<-myiris
names(iris2)[5]<-"class"

for (b in (1:B)) {
  inTrain<-createDataPartition(iris2$class,p=3/4,list=FALSE)
  trainW<-iris2[inTrain,-5]
  testW<-iris2[-inTrain,-5]
  trainwY<-iris2$class[inTrain]
  testwY<-iris2$class[-inTrain]
  fit<-caret::train(class~.,data=iris2[inTrain,],method="rpart",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=iris2[-inTrain,-5],type="raw")
  iERRMAT[b,1]<-length(pp[pp!=iris2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=iris2[inTrain,],method="ranger",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=iris2[-inTrain,-5],type="raw")
  iERRMAT[b,2]<-length(pp[pp!=iris2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=iris2[inTrain,],method="knn",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=iris2[-inTrain,-5],type="raw")
  iERRMAT[b,3]<-length(pp[pp!=iris2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=iris2[inTrain,],method="lda",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=iris2[-inTrain,-5],type="raw")
  iERRMAT[b,4]<-length(pp[pp!=iris2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=iris2[inTrain,],method="qda",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=iris2[-inTrain,-5],type="raw")
  iERRMAT[b,5]<-length(pp[pp!=iris2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=iris2[inTrain,],method="pda",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=iris2[-inTrain,-5],type="raw")
  iERRMAT[b,6]<-length(pp[pp!=iris2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=iris2[inTrain,],method="nb",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=iris2[-inTrain,-5],type="raw")
  iERRMAT[b,7]<-length(pp[pp!=iris2$class[-inTrain]])/length(pp)
  fit<-caret::train(class~.,data=iris2[inTrain,],method="mda",tuneLength=15,trControl=ctrl)
  pp<-predict(fit,newdata=iris2[-inTrain,-5],type="raw")
  iERRMAT[b,8]<-length(pp[pp!=iris2$class[-inTrain]])/length(pp)
  print(b)
}
par(mfrow=c(1,1))
boxplot(iERRMAT,names=c("CART","RF","knn","lda","qda","pda","nb","mda"))
