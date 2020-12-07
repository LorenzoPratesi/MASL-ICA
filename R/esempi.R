
#Differences between PCA and ICA

library(fastICA)
library(MASS)


#Example 1: two mixed independent uniforms

set.seed(14)

S <- matrix(runif(10000), 3000, 2)
A <- matrix(c(1, 1, -1, 3), 2, 2, byrow = TRUE)
X <- S %*% A

ica <- fastICA(X, n.comp=2, alg.typ="parallel")

par(mfrow = c(1, 3))
plot(ica$X, col="lightblue3", pch=20, main = "Original data")
plot(ica$X %*% ica$K, col="lightblue3", pch=20, main = "Principal components")
plot(ica$S, col="lightblue3", pch=20, main = "Independent components")



#Example 2: mixture of bivariate normal distributions

x1 <- mvrnorm(n = 1000, mu = c(0, 0), Sigma = matrix(c(10, 3, 3, 1), 2, 2))
x2 <- mvrnorm(n = 1000, mu = c(-1, 2), Sigma = matrix(c(10, 3, 3, 1), 2, 2))
x <- rbind(x1, x2)
par(mfrow = c(1, 3))
plot(x, col="lightblue3", pch=20, main = "Original data")
  
pca <- prcomp(x, scale=TRUE, retx=TRUE)
plot(pca$x[,1:2], col="lightblue3", pch=20, main = "Principal components")

ica <- fastICA(x, 2, alg.typ = "deflation", fun = "logcosh", alpha = 1,
               method = "R", row.norm = FALSE, maxit = 200, 
               tol = 0.0001, verbose = TRUE)
plot(ica$S, col="lightblue3", pch=20, main = "Independent components")

