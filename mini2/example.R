# DEMO LECTURE 4
#
# Relabel iris data to better see names since two of them starts with v
myiris<-iris
newfac<-rep(0,dim(myiris)[1])
newfac[iris$Species=="setosa"]<-"setosa"
newfac[iris$Species=="virginica"]<-"virginica"
newfac[iris$Species=="versicolor"]<-"ersicolor"
newfac<-as.factor(newfac)
myiris$Species<-newfac
###
library(klaR)
classscatter(Species~., data=myiris, method = "lda")
# the klaR package - you can plot classifier predictions and errors for different methods

# 3D exploration of data
library(scatterplot3d)
par(mfrow = c(2, 2))
mar0 = c(2, 3, 2, 3)
scatterplot3d(iris[, 1], iris[, 2], iris[, 3], mar = mar0, color = c("blue",
                                                                     "black", "red")[iris$Species], pch = 19)
scatterplot3d(iris[, 2], iris[, 3], iris[, 4], mar = mar0, color = c("blue",
                                                                     "black", "red")[iris$Species], pch = 19)
scatterplot3d(iris[, 3], iris[, 4], iris[, 1], mar = mar0, color = c("blue",
                                                                     "black", "red")[iris$Species], pch = 19)
scatterplot3d(iris[, 4], iris[, 1], iris[, 2], mar = mar0, color = c("blue",
                                                                     "black", "red")[iris$Species], pch = 19)

classscatter(Species~., data=myiris, method = "qda")

library(klaR)
### Exploring different methods - what do the class boundaries look like?
partimat(Species~.,data=myiris,method="lda")

partimat(Species~.,data=myiris,method="qda")

partimat(Species~.,data=myiris,method="naiveBayes")

partimat(Species~.,data=myiris,method="sknn",k=1)

partimat(Species~.,data=myiris,method="sknn",k=10)

partimat(Species~.,data=myiris,method="rpart")


### ade4 package discrimimant summary
# what do the shape of the classes look like? Which variables contribute to the leading PC components?
library(ade4)
pca1 <- dudi.pca(iris[, 1:4], scannf = FALSE ,scale=TRUE,center=TRUE)
dis1 <- discrimin(pca1, iris$Species, scannf = FALSE, nf=2)
names(dis1)
plot(dis1)


###################################
#wine data
wine<-read.table("winedata.txt")
library(scatterplot3d)
par(mfrow = c(2, 2))
mar0 = c(2, 3, 2, 3)
scatterplot3d(wine[, 2], wine[, 3], wine[, 4], mar = mar0, color = c("blue",
                                                                     "black", "red")[wine$class], pch = 19)
scatterplot3d(wine[, 3], wine[, 4], wine[, 5], mar = mar0, color = c("blue",
                                                                     "black", "red")[wine$class], pch = 19)
scatterplot3d(wine[, 4], wine[, 5], wine[, 2], mar = mar0, color = c("blue",
                                                                     "black", "red")[wine$class], pch = 19)
scatterplot3d(wine[, 5], wine[, 2], wine[, 3], mar = mar0, color = c("blue",
                                                                     "black", "red")[wine$class], pch = 19)
classscatter(class~., data=wine[,1:5], method = "lda")


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


pca1 <- dudi.pca(mywine[, 2:14], scannf = FALSE ,scale=TRUE,center=TRUE)
dis1 <- discrimin(pca1, mywine$class, scannf = FALSE)
names(dis1)
plot(dis1)


#################################################
### caret package - contains lots of different methods!
# relabel predictors to numerical
library(ElemStatLearn)
SAheart$famhist<-as.numeric(SAheart$famhist)-1

classscatter(chd~., data=SAheart, method = "lda")

library(caret)
#### training and test data
SAclass2<-SAheart
SAclass2$chd<-rep(0,dim(SAclass2)[1])
SAclass2$chd[SAheart$chd==0]<-"NoCHD"
SAclass2$chd[SAheart$chd==1]<-"CHD"



inTrain<-createDataPartition(SAclass2$chd,p=3/4,list=FALSE)
trainSA<-SAclass2[inTrain,-10]
testSA<-SAclass2[-inTrain,-10]
trainY<-SAclass2$chd[inTrain]
testY<-SAclass2$chd[-inTrain]
#

#################
par(mfrow=c(1,1))
library(pROC)
#
ctrl<-trainControl(method="repeatedcv",repeats=3,classProbs=TRUE,summaryFunction=twoClassSummary)

ldafit<-train(chd~.,data=SAclass2[inTrain,],method="lda",tuneLength=15,metric="ROC",trControl=ctrl)
pp<-predict(ldafit,newdata=SAclass2[-inTrain,-10],type="prob")
p2<-SAclass2$chd[-inTrain]
p2[pp[,1]>.5]<-"CHD"
p2[pp[,1]<.5]<-"NoCHD"
print(length(p2[p2!=SAclass2[-inTrain,10]])/length(p2))
rr<-roc(SAclass2$chd[-inTrain],pp[,2])
plot(rr)
# comparing methods based on ROC - true positive rate vs 1-false positive rate
# Specificity = True negative rate (1-False positive)
# Sensitivity = True positive rate

knnfit<-train(chd~.,data=SAclass2[inTrain,],method="knn",tuneLength=25,metric="ROC",trControl=ctrl)
pp<-predict(knnfit,newdata=SAclass2[-inTrain,-10],type="prob")
p2<-SAclass2$chd[-inTrain]
p2[pp[,1]>.5]<-"CHD"
p2[pp[,1]<.5]<-"NoCHD"
print(length(p2[p2!=SAclass2[-inTrain,10]])/length(p2))
rr<-roc(SAclass2$chd[-inTrain],pp[,2])
plot(rr,add=TRUE,col=2)

qdafit<-train(chd~.,data=SAclass2[inTrain,],method="qda",tuneLength=15,metric="ROC",trControl=ctrl)
pp<-predict(qdafit,newdata=SAclass2[-inTrain,-10],type="prob")
p2<-SAclass2$chd[-inTrain]
p2[pp[,1]>.5]<-"CHD"
p2[pp[,1]<.5]<-"NoCHD"
print(length(p2[p2!=SAclass2[-inTrain,10]])/length(p2))
rr<-roc(SAclass2$chd[-inTrain],pp[,2])
plot(rr,add=TRUE,col=3)



nbfit<-train(chd~.,data=SAclass2[inTrain,],method="nb",tuneLength=15,metric="ROC",trControl=ctrl)
pp<-predict(nbfit,newdata=SAclass2[-inTrain,-10],type="prob")
p2<-SAclass2$chd[-inTrain]
p2[pp[,1]>.5]<-"CHD"
p2[pp[,1]<.5]<-"NoCHD"
print(length(p2[p2!=SAclass2[-inTrain,10]])/length(p2))
rr<-roc(SAclass2$chd[-inTrain],pp[,2])
plot(rr,add=TRUE,col=4)

rpartfit<-train(chd~.,data=SAclass2[inTrain,],method="rpart",tuneLength=15,metric="ROC",trControl=ctrl)
pp<-predict(rpartfit,newdata=SAclass2[-inTrain,-10],type="prob")
p2<-SAclass2$chd[-inTrain]
p2[pp[,1]>.5]<-"CHD"
p2[pp[,1]<.5]<-"NoCHD"
print(length(p2[p2!=SAclass2[-inTrain,10]])/length(p2))
rr<-roc(SAclass2$chd[-inTrain],pp[,2])
plot(rr,add=TRUE,col=5)


rffit<-train(chd~.,data=SAclass2[inTrain,],method="ranger",tuneLength=15,metric="ROC",trControl=ctrl)
pp<-predict(rffit,newdata=SAclass2[-inTrain,-10],type="prob")
p2<-SAclass2$chd[-inTrain]
p2[pp[,1]>.5]<-"CHD"
p2[pp[,1]<.5]<-"NoCHD"
print(length(p2[p2!=SAclass2[-inTrain,10]])/length(p2))
rr<-roc(SAclass2$chd[-inTrain],pp[,2])
plot(rr,add=TRUE,col=6)



pdafit<-train(chd~.,data=SAclass2[inTrain,],method="pda",tuneLength=15,metric="ROC",trControl=ctrl)
pp<-predict(pdafit,newdata=SAclass2[-inTrain,-10],type="prob")
p2<-SAclass2$chd[-inTrain]
p2[pp[,1]>.5]<-"CHD"
p2[pp[,1]<.5]<-"NoCHD"
print(length(p2[p2!=SAclass2[-inTrain,10]])/length(p2))
rr<-roc(SAclass2$chd[-inTrain],pp[,2])
plot(rr,add=TRUE,col=2,lty=2)


names(getModelInfo())


####
plot(knnfit)

plot(rpartfit)

plot(rffit)

plot(pdafit)


#### If you want to apply caret to multiclass data, the classes have to have LETTER names like "A", "B", "C" and 
#### you have to change to multiclassSummary and use a different metric like Accuracy instead of ROC.
wine2<-wine
wine2$class[wine$class==1]<-"A"
wine2$class[wine$class==2]<-"B"
wine2$class[wine$class==3]<-"C"


inTrain<-createDataPartition(wine2$class,p=3/4,list=FALSE)
trainW<-wine2[inTrain,-1]
testW<-wine2[-inTrain,-1]
trainwY<-wine2$class[inTrain]
testwY<-wine2$class[-inTrain]


#################
#
ctrl<-trainControl(method="repeatedcv",repeats=3,summaryFunction=multiClassSummary)

rpartfit<-train(class~.,data=wine2[inTrain,],method="rpart",tuneLength=15,trControl=ctrl)
plot(rpartfit)
pp<-predict(rpartfit,newdata=wine2[-inTrain,-1],type="raw")
table(pp,wine2$class[-inTrain])
#####