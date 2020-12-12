
# Minimum distance index
# The MD index is scaled in such a way, that it takes a value between 0 and 1. And 0 corresponds to an optimal separation.

library(JADE)

S <- cbind(rt(1000, 4), rnorm(1000), runif(1000))
A <- matrix(rnorm(9), ncol = 3) 
X <- S %*% t(A)

jade <- JADE(X,3)

W.hat <- jade$W # estimated unmixing matrix
MD(W.hat, A)

par(mfrow=c(3,3))
plot(S[,1], col="lightblue3", pch=16, main="S1")
plot(S[,2], col="lightblue3", pch=16, main="S2")
plot(S[,3], col="lightblue3", pch=16, main="S3")

plot(X[,1], col="lightblue3", pch=16, main="X1")
plot(X[,2], col="lightblue3", pch=16, main="X2")
plot(X[,3], col="lightblue3", pch=16, main="X3")

plot(jade$S[,1], col="lightblue3", pch=16, main="estimated S1")
plot(jade$S[,2], col="lightblue3", pch=16, main="estimated S2")
plot(jade$S[,3], col="lightblue3", pch=16, main="estimated S3")

############################

#Tests for the dimension

library(ICtest)

set.seed(1)

n <- 1450
S <- cbind(runif(n), rchisq(n, 2), rexp(n), rnorm(n), rnorm(n), rnorm(n)) # 3 components and 3 noises
A <- matrix(rnorm(36), ncol = 6)
X <- S  %*% t(A)

FOBIasymp(X, k = 2, type = "S1", model = "ICA")
#there aren't 4 or more gaussian components

FOBIasymp(X, k = 3, type = "S1", model = "ICA")
#there are 3 or more gaussian components


FOBIasymp(X, k = 2, type = "S2", model = "ICA")
FOBIasymp(X, k = 3, type = "S2", model = "ICA")


test <- FOBIasymp(X, k = 3, type = "S3", model = "ICA")
ggscreeplot(test) + geom_hline(yintercept=8) # p+2=8
test
# The screeplot shows that the components are in the right order and 3 components have an eigenvalue close to the value of interest.

FOBIboot(X, k = 3, s.boot = "B1")

FOBIboot(X, k = 3, s.boot = "B2")
