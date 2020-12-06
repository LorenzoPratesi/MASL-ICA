library("JADE")
library("tuneR")

S1 <- readWave(system.file("datafiles/source5.wav", package = "JADE"))
S2 <- readWave(system.file("datafiles/source7.wav", package = "JADE"))
S3 <- readWave(system.file("datafiles/source9.wav", package = "JADE"))

set.seed(321)
NOISE <- noise("white", duration = 50000)
S <- cbind(S1@left, S2@left, S3@left, NOISE@left)
S <- scale(S, center = FALSE, scale = apply(S, 2, sd))
St <- ts(S, start = 0, frequency = 8000)
p <- 4
A <- matrix(runif(p^2, 0, 1), p, p)
A

X <- tcrossprod(St, A)
Xt <- as.ts(X)

plot(St, main = "Sources")
plot(Xt, main = "Mixtures")

x1 <- normalize(Wave(left = X[, 1], samp.rate = 8000, bit = 8), unit = "8")
x2 <- normalize(Wave(left = X[, 2], samp.rate = 8000, bit = 8), unit = "8")
x3 <- normalize(Wave(left = X[, 3], samp.rate = 8000, bit = 8), unit = "8")
x4 <- normalize(Wave(left = X[, 4], samp.rate = 8000, bit = 8), unit = "8")

setWavPlayer('/usr/bin/afplay')

play(x1)
play(x2)
play(x3)
play(x4)


jade <- JADE(X)
sobi <- SOBI(Xt)
nsstdjd <- NSS.TD.JD(Xt)

sobi

sobi2 <- SOBI(Xt, k = c(1, 2, 5, 10, 20))
round(coef(sobi) %*% A, 4)

c(jade = MD(coef(jade), A), sobi = MD(coef(sobi), A), sobi2 = MD(coef(sobi2), A), nsstdjd = MD(coef(nsstdjd), A))

Z.nsstdjd <- bss.components(nsstdjd)

NSSTDJDwave1 <- normalize(Wave(left = as.numeric(Z.nsstdjd[, 1]), samp.rate = 8000, bit = 8), unit = "8")

NSSTDJDwave2 <- normalize(Wave(left = as.numeric(Z.nsstdjd[, 2]), samp.rate = 8000, bit = 8), unit = "8")

NSSTDJDwave3 <- normalize(Wave(left = as.numeric(Z.nsstdjd[, 3]), samp.rate = 8000, bit = 8), unit = "8")

NSSTDJDwave4 <- normalize(Wave(left = as.numeric(Z.nsstdjd[, 4]), samp.rate = 8000, bit = 8), unit = "8")


play(NSSTDJDwave1)
play(NSSTDJDwave2)
play(NSSTDJDwave3)
play(NSSTDJDwave4)



