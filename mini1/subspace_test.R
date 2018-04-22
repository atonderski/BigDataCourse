# Read data with two clusters (2d)
library(R.matlab)
library(subspace)

set.seed(13)
require(mnormt)
varcov <- diag(2)*0.2
data_new <- rmnorm(500, mean=c(0,0), varcov=varcov)
data_new <- rbind(data_new,rmnorm(500, mean=c(3,3), varcov=varcov))
data_new <- rbind(data_new,rmnorm(500, mean=c(6,6), varcov=varcov))

# c_data <- readMat("/Users/adam/Programming/BigData/matlab-3clust.mat")[["data"]]
c_data <- data_new
# Add extra column, as unrelated feature
new_col <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col)
new_col1 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col1)
new_col2 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col2)
new_col3 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col3)
new_col4 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col4)
new_col5 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col5)
new_col6 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col6)
new_col7 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col7)
new_col8 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col8)
new_col9 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col9)
new_col10 <- runif(nrow(c_data), -200, 200)
c_data <- cbind(c_data, new_col10)
c_data <- sweep(c_data,2,apply(c_data,2,min))
c_data <- scale(c_data, center=FALSE, scale=apply(c_data,2,max))

# Add randomness in all dimensions
for (i in 1:100){
  c_data <- rbind(c_data, runif(ncol(c_data), 0, 1))
}

data <- as.data.frame(c_data)

# clean_data = readRDS("clean_data.rds")
# noisy_data = readRDS("noisy_data.rds")

# clustering <- FIRES(data, base_dbscan_epsilon=0.02, base_dbscan_minpts=100, post_dbscan_epsilon = 0.1, post_dbscan_minpts = 50)
# clustering <- CLIQUE(data, 10, 0.2)
# clustering <- SubClu(data, epsilon = 0.1, minSupport = 200)
clustering <- P3C(data, 0.00001)
plot(clustering, data)

get_dims <- function(cluster){
  return(sum(cluster[['subspace']]))
}
new_clustering = clustering[lapply(clustering, get_dims) > 1]
plot(new_clustering, data)

# library("rgl")
# plot3d(c_data, xlab = "x", ylab = "y", zlab = "z")

library("cluster")
library("dbscan")
library("fpc")
kk <- dbscan(data[,1], 0.05, 200)
kk
plot(data[,1], col=kk$cluster+1, main="dbscan", xlab="x", ylab="y")
# plotcluster(data[,1:2], kk$cluster)
# clusplot(data, kk$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)


