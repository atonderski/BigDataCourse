library(MASS)
library(GGally)

# from first
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(gplots)
library(lattice)
library(svdvis)
library(softImpute)
library(ElemStatLearn)
library(rpart.plot)
library(randomForest)
library(ranger)
library(rgl)
library(irlba)
library(bigmemory)
library(biglm)
library(biglars)

data(iris)
ggpairs(data=iris, # data.frame with variables
        columns=1:4, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color = Species))
# Run kmeans a couple of times ... kmeans starts from a random partition - can fail!
# random starting point is used..
kk<-kmeans(iris[,1:4],3)
table(kk$cluster,iris$Species) # Check if the cluster indices overal with the true species labels
#
mydata<-iris
mydata$km<-as.factor(kk$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:4, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color = Species, shape = km))
#
ll<-lda(iris[,1:4],iris$Species)
pl<-predict(ll,iris[,1:4])
table(pl$class,iris$Species)
# Supervised (classification) can predict the labels - class discovery (clustering) is a more difficult problem!
mydata[,1:4]<-scale(mydata[,1:4])
kk<-kmeans(iris[,1:4],3)
table(kk$cluster,iris$Species) # Check if the cluster indices overal with the true species labels
#
mydata$km<-as.factor(kk$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:4, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color = Species, shape = km)
)
#
#########################################
# How many clusters in the data?
# Run for K=1 to 10 and summarize the deviation from the cluster means
# The so-called Within-Sum-of-Squares
data(iris)
W<-rep(0,10)
for (k in 2:10) {
  kk<-kmeans(iris[,1:4],k)
  W[k]<-kk$tot.with }
mu<-apply(iris[,1:4],2,mean)
iriss<-iris[,1:4]-t(matrix(rep(mu,dim(iris)[1]),4,dim(iris)[1]))
W[1]<-sum(iriss^2)
plot(seq(1,10),W,xlab="K",ylab="Within-SS",type="b")
# This looks a bit like the RSS-curve in regression. 
# Remember what to look for? When does the curve level off? 
# When does it stop paying off to add clusters?
############# 
library(cluster) 
#########################
# PAM - partion around mediods - a robust version of kmeans
pp<-pam(iris[,1:4],3)
plot(pp)
# Are the 3 clusters well separated?
pairs(iris[,1:4],col=as.numeric(iris$Species)+1,pch=pp$cluster)
#
clusplot(iris[,1:4], pp$cluster, color=TRUE, shade=TRUE, 
         labels=2, lines=0)

#
W<-rep(0,10)
for (k in 2:10) {
  kk<-pam(dist(iris[,1:4]),k)
  W[k]<-kk$sil$avg }
mu<-apply(iris[,1:4],2,mean)
iriss<-iris[,1:4]-t(matrix(rep(mu,dim(iris)[1]),4,dim(iris)[1]))
plot(seq(2,10),W[2:10],xlab="K",ylab="Silhouette Width",type="l")
# For which K is the silhouette width maximized?
##############################
# hierarchical clustering
hh<-hclust(dist(iris[,1:4]))
plot(hh,label=iris$Species)
table(cutree(hh,k=3),iris$Species)
# pretty good separation
# Try a different joining mechanism in the tree
hh<-hclust(dist(iris[,1:4]),"single")
plot(hh,label=iris$Species)
table(cutree(hh,k=3),iris$Species)
#
hh<-hclust(dist(iris[,1:4]),"ward.D")
plot(hh,label=iris$Species)
table(cutree(hh,k=3),iris$Species)
# Big impact! You get what you ask for in clustering...
cc<-cor(t(iris[,1:4]))
hh<-hclust(as.dist(1-cc))
plot(hh,label=iris$Species)
table(cutree(hh,k=3),iris$Species)
# Also - how you measure distance between the observations also has an impact.
# Choices: distance metric AND joining meachnism (linkage)
##########################################################
# Let's look at the numbers data
library(ElemStatLearn)
data(zip.train)
Numbers<-as.data.frame(zip.train)
Numbers[,1]<-as.factor(Numbers[,1])
names(Numbers)<-c("number",as.character(seq(1,256)))
Nmat<-Numbers[,-1]
N<-2000
iu<-sample(seq(1,7291),N)
# N digits chosen at random
kk<-kmeans(Nmat[iu,],10)
table(kk$cluster,Numbers[iu,1])
# Check if the cluster indices overal with the true species labels
#
# How many clusters in the data?
# Run for K=1 to 10 and summarize the deviation from the cluster means
# The so-called Within-Sum-of-Squares
W<-rep(0,25)
for (k in 2:25) {
  kk<-kmeans(Nmat[iu,],k)
  W[k]<-kk$tot.with }
mu<-apply(Nmat[iu,],2,mean)
mss<-Nmat[iu,]-t(matrix(rep(mu,dim(Nmat[iu,])[1]),256,dim(Nmat[iu,])[1]))
W[1]<-sum(mss^2)
plot(seq(1,25),W,xlab="K",ylab="Within-SS",type="l")
### Clustering after PCA dimension reduction
ssn<-svd(Nmat[iu,])
plot3d(ssn$u[,1:3],col=as.numeric(Numbers[iu,1]))
#
W<-rep(0,25)
for (k in 2:25) {
  kk<-kmeans(ssn$u[,1:10],k)
  W[k]<-kk$tot.with }
mu<-apply(ssn$u[,1:10],2,mean)
mss<-ssn$u[,1:10]-t(matrix(rep(mu,dim(ssn$u[,1:10])[1]),10,dim(ssn$u[,1:10])[1]))
W[1]<-sum(mss^2)
plot(seq(1,25),W,xlab="K",ylab="Within-SS",type="l")
############# 
# PAM - partion around mediods - a robust version of kmeans
pp<-pam(Nmat[iu,],4)
plot(pp)
#
plot(pp$silinfo$widths[,3],col=pp$silinfo$widths[,1],type="h")
points(seq(1,N),rep(-0.1,N),pch=as.character((Numbers[iu,1])[as.numeric(rownames(pp$silinfo$widths))]))
# Looks like an OK cluster separation - but are the digits in the clusters?
table(pp$clustering,Numbers[iu,1])
#
pp<-pam(ssn$u[,1:15],4)
plot(pp$silinfo$widths[,3],col=pp$silinfo$widths[,1],type="h")
points(seq(1,N),rep(-0.1,N),pch=as.character((Numbers[iu,1])[as.numeric(rownames(pp$silinfo$widths))]))
# Looks like an OK cluster separation - but are the digits in the clusters?
table(pp$clustering,Numbers[iu,1])
# Not great!!!
##############################
# hierarchical clustering
hh<-hclust(dist(Nmat[iu,]),"complete")
plot(hh,label=Numbers[iu,1])
# 
hh<-hclust(dist(Nmat[iu,]),"single")
plot(hh,label=Numbers[iu,1])
# Single linkage aggressively builds the 1s as a cluster first
hh<-hclust(dist(Nmat[iu,]),"average")
plot(hh,label=Numbers[iu,1])
#
cc<-cor(t(Nmat[iu,]))
hh<-hclust(as.dist(1-cc))
plot(hh,label=Numbers[iu,1])
#####################
library(mclust)
kk<-Mclust(iris[,1:4])
table(kk$class,iris$Species)
summary(kk)
plot(kk) # try density, BIC
#
kk<-Mclust(iris[,1:4],model="EEE")
table(kk$class,iris$Species)
summary(kk)
plot(kk) # try density, BIC
#
kk<-Mclust(iris[,1:4],G=3)
table(kk$class,iris$Species)
summary(kk)
plot(kk) # try density, BIC
###########################
kk<-Mclust(Nmat[iu,]) # REALLY SLOW!!!
table(kk$class,Numbers[iu,1])
#
ssn<-svd(Nmat[iu,])
kk<-Mclust(ssn$u[,1:15]) # reduced dimension
table(kk$class,Numbers[iu,1])
plot(kk)
# Not so easy....
##############################
install.packages("dbscan")
library(dbscan)
##
db<-dbscan(iris[,1:4], eps=1, minPts = 10)
table(db$cluster,iris$Species)
#
db<-dbscan(iris[,1:4], eps=.5, minPts = 10)
table(db$cluster,iris$Species)
#
db<-dbscan(iris[,1:4], eps=2, minPts = 5)
table(db$cluster,iris$Species)
# tuning parameters eps and minPts + choice of distance
kNNdistplot(iris[,1:4], k=10) # eps .5?
#
db<-dbscan(iris[,1:4], eps=.5, minPts = 2)
table(db$cluster,iris$Species)
#
mydata<-iris
mydata$cl<-as.factor(db$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:4, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color = Species, shape=cl))
#
par(mfrow=c(1,1))
hullplot(iris[,1:4],db)
## other types of distances
db<-dbscan(dist(apply(iris[,1:4],2,scale),"manhattan"), eps=1, minPts = 15)
table(db$cluster,iris$Species)
#
mydata<-iris
mydata$cl<-as.factor(db$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:4, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color = Species, shape=cl))
# 
library(kernlab)
data(spirals)
spirs<-as.data.frame(spirals)
names(spirs)<-c("x","y")
ggpairs(data=spirs, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="spirals data", # title of the plot
)
#
kNNdistplot(spirs,k=5) #.17?
#
db<-dbscan(spirs[,1:2], eps=.17, minPts = 5)
mydata<-spirs
mydata$cl<-as.factor(db$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color =cl))

# sensitive to choices...
db<-dbscan(spirs[,1:2], eps=.17, minPts = 10, borderPoints = T)
mydata<-spirs
mydata$cl<-as.factor(db$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color =cl))
#
db<-dbscan(spirs[,1:2], eps=.2, minPts = 5, borderPoints = T)
mydata<-spirs
mydata$cl<-as.factor(db$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color =cl))
#
kk<-kmeans(spirs[,1:2],2)
mydata<-spirs
mydata$cl<-as.factor(kk$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color =cl))
#
kk<-cutree(hclust(dist(spirs[,1:2])),k=2)
mydata<-spirs
mydata$cl<-as.factor(kk)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color =cl))
#
kk<-cutree(hclust(dist(spirs[,1:2])),k=10)
mydata<-spirs
mydata$cl<-as.factor(kk)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color =cl))
####################
db<-hdbscan(spirs[,1:2], minPts = 8)
plot(db$hc)
db
#
mydata<-spirs
mydata$cl<-as.factor(db$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Iris Data", # title of the plot
        mapping = ggplot2::aes(color =cl))
###
data(moons)
ggpairs(data=moons, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Moons", # title of the plot
)
#
kNNdistplot(moons[,1:2],k=5) # .4?
#
db<-dbscan(moons[,1:2], eps=.4, minPts = 5)
mydata<-moons
mydata$cl<-as.factor(db$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Moons", # title of the plot
        mapping = ggplot2::aes(color =cl))
#
db<-dbscan(moons[,1:2], eps=.5, minPts = 5)
mydata<-moons
mydata$cl<-as.factor(db$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Moons", # title of the plot
        mapping = ggplot2::aes(color =cl))
#
db<-hdbscan(moons[,1:2], minPts = 5)
plot(db$hc)
db
#
mydata<-moons
mydata$cl<-as.factor(db$cluster)
ggpairs(data=mydata, # data.frame with variables
        columns=1:2, # columns to plot, default to all.
        title="Moons", # title of the plot
        mapping = ggplot2::aes(color =cl))
# some nice plotting functions
db<-dbscan(moons[,1:2], eps=.5, minPts = 5)
library("factoextra")
fviz_cluster(db, moons, stand = FALSE, frame = FALSE, geom = "point")
########
ssn<-svd(Nmat[iu,])
kNNdistplot(ssn$u[,1:6],k=10) # eps 0.02?
# 
db<-dbscan(ssn$u[,1:6], eps=.015, minPts = 10)
db
table(db$clust,Numbers[iu,1])
#
kNNdistplot(ssn$u[,1:25],k=10) #
# 
db<-dbscan(ssn$u[,1:25], eps=.04, minPts = 10)
db
table(db$clust,Numbers[iu,1])
#

