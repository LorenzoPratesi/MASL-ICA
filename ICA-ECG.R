# Title     : TODO
# Objective : TODO
# Created by: lorenzopratesi
# Created on: 05/12/20

library("JADE")
library("BSSasymp")
dataset <- matrix(scan("resources/foetal_ecg.dat"), 2500, 9, byrow = TRUE)

X <- dataset[, 2:9]
plot.ts(X, nc = 1, main = "Data")

scale(X, center = FALSE, scale = apply(X, 2, sd))
jade <- JADE(X)
plot.ts(bss.components(jade), nc = 1, main = "JADE solution")

ascov <- ASCOV_JADE_est(X)
Vars <- matrix(diag(ascov$COV_W), nrow = 8)
Coefs <- coef(jade)[4, ]
SDs <- sqrt(Vars[4, ])
Coefs

SDs

w <- as.vector(coef(jade))
V <- ascov$COV_W
L1 <- L2 <- L3 <- rep(0, 64)
L1[5 * 8 + 4] <- L2[6 * 8 + 4] <- L3[7 * 8 + 4] <- 1
L <- rbind(L1, L2, L3)
Lw <- L %*% w
T <- t(Lw) %*% solve(L %*% tcrossprod(V, L), Lw)
T

format.pval(1 - pchisq(T, 3))