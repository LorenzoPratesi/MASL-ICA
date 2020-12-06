

library(fastICA)

set.seed(14)


S <- matrix(runif(10000), 5000, 2)
A <- matrix(c(1, 1, -1, 3), 2, 2, byrow = TRUE)
X <- S %*% A

ica <- fastICA(X, n.comp=2, alg.typ="parallel")
par(mfrow = c(1, 3))
plot(ica$X, col="lightblue3", pch=20, main = "Original data")
plot(ica$X %*% ica$K, col="lightblue3", pch=20, main = "Principal components")
plot(ica$S, col="lightblue3", pch=20, main = "Independent components")
