#### Install packages ####
list.of.packages <- c("mlr", "unbalanced", "caret", "ROSE", "DMwR")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

#### Sampling methods ####
## Classification
library(caret)
evaluateClassifiers <- function(dd, nIterations) {
  
  trainParams <- trainControl(method="cv", summaryFunction=multiClassSummary)
  
  methods <- c("ranger", "knn", "lda", "qda")
  readableMethods <- c("RF","kNN","LDA","QDA")
  
  sampleMethods <- c("raw", "up-random", "up-smote", "down-random", "down-tomek")
  
  kappaMatrix <- matrix(0, length(sampleMethods), length(methods))
  for (iIteration in 1:nIterations) {
    # Create partitioning indexes for testing
    inTrain = createDataPartition(dd$class, p=3/4, list=FALSE)
    trainData = dd[inTrain,]
    testData = dd[-inTrain,]
    testClasses = testData$class
    testData[, c("class")] = list(NULL) # Test data without true classes
    
    for(iSampleMethod in 1:length(sampleMethods)) {
      trainData = createSample(trainData, sampleMethods[iSampleMethod])
      
      for(iMethod in 1:length(methods)) {
        fit <- caret::train(class~., data=trainData, method=methods[iMethod], tuneLength=15, trControl=trainParams)
        pp <- predict(fit, newdata=testData, type="raw")
        conf <- confusionMatrix(pp, testClasses)
        kappa <- conf$overall[2]
        kappaMatrix[iSampleMethod, iMethod] = kappaMatrix[iSampleMethod, iMethod] + kappa
      }
    }
  }
  
  kappaMatrix = kappaMatrix / nIterations
  
  return(list("classMethods" = readableMethods, "sampleMethods" = sampleMethods, "data" = kappaMatrix))
}

library(DMwR)
library(ROSE)
evaluateClassifiers <- function(dd, nIterations) {
  
  trainParams <- trainControl(method="cv", summaryFunction=multiClassSummary)
  
  methods <- c("ranger", "knn", "lda", "qda")
  readableMethods <- c("RF","kNN","LDA","QDA")
  
  sampleMethods <- c("none", "up", "smote", "down", "rose")
  readableSampleMethods <- c("raw", "up-random", "up-smote", "down-random", "up-rose")
  
  kappaMatrix <- matrix(0, length(sampleMethods), length(methods))
  confusion = list("classMethod"= NULL, "sampleMethod" = NULL, "conf" = NULL)
  maxKappa = -1
  for (iIteration in 1:nIterations) {
    # Create partitioning indexes for testing
    inTrain = createDataPartition(dd$class, p=3/4, list=FALSE)
    trainData = dd[inTrain,]
    testData = dd[-inTrain,]
    testClasses = testData$class
    testData[, c("class")] = list(NULL) # Test data without true classes
    
    for(iSampleMethod in 1:length(sampleMethods)) {
      if (sampleMethods[iSampleMethod] == "none")
        trainParams$sampling <- NULL
      else
        trainParams$sampling <- sampleMethods[iSampleMethod]
      
      for(iMethod in 1:length(methods)) {
        fit <- caret::train(class~., data=trainData, method=methods[iMethod], tuneLength=15, trControl=trainParams)
        pp <- predict(fit, newdata=testData, type="raw")
        conf <- confusionMatrix(pp, testClasses)
        kappa <- conf$overall[2]
        kappaMatrix[iSampleMethod, iMethod] = kappaMatrix[iSampleMethod, iMethod] + kappa
        
        if (kappa > maxKappa) {
          maxKappa = kappa
          confusion$classMethod = methods[iMethod]
          confusion$sampleMethod = sampleMethods[iSampleMethod]
          confusion$conf = conf
        }
      }
    }
  }
  
  kappaMatrix = kappaMatrix / nIterations
  
  return(list("classMethods" = readableMethods, "sampleMethods" = readableSampleMethods, "data" = kappaMatrix, "confusion" = confusion))
}

classify <- function(dd, classMethod, sampleMethod=NULL) {
  trainParams <- trainControl(method="cv", summaryFunction=multiClassSummary)
  if (!is.null(sampleMethod))
    trainParams$sampling = sampleMethod
    
  # Create partitioning indexes for testing
  inTrain = createDataPartition(dd$class, p=3/4, list=FALSE)
  trainData = dd[inTrain,]
  testData = dd[-inTrain,]
  testClasses = testData$class
  testData[, c("class")] = list(NULL) # Test data without true classes
  
  fit <- caret::train(class~., data=trainData, method=classMethod, tuneLength=15, trControl=trainParams)
  pp <- predict(fit, newdata=testData, type="raw")
  conf <- confusionMatrix(pp, testClasses)
  
  return(conf)
}

evaluateClassifiers2 <- function(dd, nIterations) {
  
  trainParams <- trainControl(method="cv", summaryFunction=multiClassSummary)
  
  methods <- c("rpart", "ranger", "knn", "lda", "qda", "pda", "nb", "mda")
  readableMethods <- c("CART","RF","kNN","LDA","QDA","PDA","NB","MDA")
  
  ratioMatrix <- matrix(0, nIterations, length(methods))
  for (iIteration in 1:nIterations) {
    # Create partitioning indexes for testing
    inTrain <- createDataPartition(dd$class, p=3/4, list=FALSE)
    trainData <- dd[inTrain,]
    testData <- dd[-inTrain,]
    testData[, c("class")] <- list(NULL) # Test data without true classes
    testClasses <- dd$class[-inTrain]
    
    for(iMethod in 1:length(methods)) {
      fit <- caret::train(class~., data=trainData, method=methods[iMethod], tuneLength=15, trControl=trainParams)
      pp <- predict(fit, newdata=testData, type="raw")
      nErrors <- length(pp[pp != testClasses])
      ratioMatrix[iIteration, iMethod] <- nErrors / length(pp) # Error ratio of predictions
    }
  }
  
  return(list("methods" = readableMethods, "data" = ratioMatrix))
}

library(unbalanced)
createSample <- function(dd, method) {
  ddLabels = dd$class
  ddData = dd
  ddData[, c("class")] = list(NULL) # Test data without true classes
  
  if (method == "up-random")
    res = ubOver(X=ddData, Y=ddLabels)
  else if (method == "up-smote")
    res = ubSMOTE(X=ddData, Y=ddLabels)
  else if (method == "down-random")
    res = ubUnder(X=ddData, Y=ddLabels)
  else if (method == "down-tomek")
    res = ubTomek(X=ddData, Y=ddLabels)
  else if (method == "raw")
    res = list("X" = ddData, "Y" = ddLabels)
  else
    stop(paste("Unsupported sampling method: ", method, sep=""))
  
  out = cbind(res$X, res$Y)
  names(out)[ncol(out)]<-"class"
  return(out)
}

#### Data analysis ####

# Relative self esteem data
setwd('~/Programming/BigData/mini3')
dat = read.csv("RSE.csv", header = TRUE, sep = "\t")
dat$age[dat$age < 20] = 1
dat$age[dat$age >= 20 & dat$age < 40] = 2
dat$age[dat$age >= 40] = 3
dat$age <- factor(dat$age)
names(dat)[12]<-"class"
dat = dat[-c(11, 13, 14)]
class_ind = 11
subdat = dat[sample(1:nrow(dat), 2000, replace=FALSE),]

# Affair data
dat = read.csv("C:/Users/jcber/Documents/Github/cas/cas-courses/msa220/3-analysis/affairs.csv", header = TRUE)
dat$nbaffairs <- dat$nbaffairs > 0
dat$nbaffairs <- factor(dat$nbaffairs)
names(dat)[9]<-"class"
class_ind = 9
subdat = dat

# Plot it
featurePlot(subdat[,-class_ind], subdat$class, "pairs")
counts <- table(subdat$class)
barplot(counts)

# Old classification with box plots
datDown <- downSample(dat[-class_ind], dat$class)
datUp <- upSample(dat[-class_ind], dat$class)
names(datDown)[class_ind]<-"class"
names(datUp)[class_ind]<-"class"
res <- evaluateClassifiers(dat, 3)
resDown <- evaluateClassifiers2(datDown, 3)
resUp <- evaluateClassifiers2(datUp, 3)
par(mfrow = c(3, 1))
boxplot(res$data, names=res$methods, main="Raw", ylab="Kappa")
boxplot(resDown$data, names=resDown$methods, main="Downsampled", ylab="Kappa")
boxplot(resUp$data, names=resUp$methods, main="Upsampled", ylab="Kappa")

# New with creation of line plot
res <- evaluateClassifiers(dat, 3)
dimnames(res$data) <- list(res$sampleMethods, res$classMethods)
res$data = t(res$data)
matplot(res$data, type = c("o"), pch=1, col = 1:5, xaxt = "n", ylab="Kappa") #plot
legend("bottomright", legend = res$sampleMethods, col=1:5, pch=1) # optional legend
axis(1, at=1:4, labels=res$classMethods[1:4])

# Single classification
minidat = dat[sample(1:nrow(dat), 1000, replace=FALSE),]
classify(minidat, 'lda', 'smote')

## Happiness data
hap = read.csv("C:/Users/jcber/Documents/Github/cas/cas-courses/msa220/3-analysis/happiness2017.csv", header = TRUE)
hap = hap[-1:-5,] # Remove to have 150 obs
# Make into 3 class problem
hap[hap[,2] < 51, 2] = 1
hap[hap[,2] > 100, 2] = 3
hap[hap[,2] > 50, 2] = 2
names(hap)[2]<-"class"
hap$class <- factor(hap$class)
hap = hap[,-3:-5]
hap = hap[,-1]
featurePlot(hap[,-2], hap$class, "pairs", auto.key = list(columns = 3))

# New with creation of line plot
hap = hap[-1:-20,]
hap_conf = classify(hap, "ranger")
hap_conf2 = classify(hap, "ranger", "smote")

## Beer data
mc = read.csv("C:/Users/jcber/Documents/Github/cas/cas-courses/msa220/3-analysis/menu.csv", header = TRUE)

mc$Category[mc$Category != "Breakfast"] = "0"
mc$Category[mc$Category == "Breakfast"] = "1"

featurePlot(hap[,-2], hap$class, "pairs", auto.key = list(columns = 3))

# New with creation of line plot
hap = hap[-1:-20,]
hap_conf = classify(hap, "ranger")
hap_conf2 = classify(hap, "ranger", "smote")

# experiment with roc plot
tpr = matrix(,3,2)
fpr = matrix(,3,2)
tpr[1,1] = 0.6
tpr[1,2] = 0.7
tpr[2,1] = 0.8
tpr[2,2] = 0.6
tpr[3,1] = 0.4
tpr[3,2] = 0.75
fpr[1,1] = 0.4
fpr[1,2] = 0.8
fpr[2,1] = 0.5
fpr[2,2] = 0.9
fpr[3,1] = 0.1
fpr[3,2] = 0.9
par(mar=c(2.5, 2.5, 2.5, 2.5))
plot(c(0,1), c(0,1), type='l', lwd=2, lty=2, col='red', xlim=c(0.03, 0.97), ylim=c(0.03, 0.97))
title(main="ROC Space")

point_colors = c("blue", "purple","forestgreen")
method_names = c("lda", "ranger", "qda")

point_characters <- c(1,2)
sample_names = c("raw", "up")

points(tpr, fpr, cex=2, lwd=2, pch=rep(point_characters, length(point_colors)), col=t(rep(point_colors, length(point_characters))))
legend("topleft", method_names, cex=1.2, col=point_colors, lwd=2, bty="n")
legend("bottomright", sample_names, cex=1.2, pch=point_characters, bty="n")

apply(rep(point))