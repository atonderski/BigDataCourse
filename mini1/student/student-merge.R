d1=read.table("~/Programming/BigData/student/student-mat.csv",sep=";",header=TRUE)
# d2=read.table("~/Programming/BigData/student/student-por.csv",sep=";",header=TRUE)

# d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))

d3 = d1
indx <- sapply(d3, is.factor)
d3[indx] <- lapply(d3[indx], function(x) as.numeric(x))
d3 <- sweep(d3,2,apply(d3,2,min))
d3 <- scale(d3, center=FALSE, scale=apply(d3,2,max))

library(subspace)
clustering <- P3C(d3, 0.01)
plot(clustering, d3)


