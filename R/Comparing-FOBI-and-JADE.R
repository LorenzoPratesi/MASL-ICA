library("JADE")
library("BSSasymp")

ICAsim <- function(ns, repet) {
  M <- length(ns) * repet
  MD.fobi <- numeric(M)
  MD.jade <- numeric(M)
  A <- diag(3)
  row <- 0
  for (j in ns) {
    for (i in 1:repet) {
      row <- row + 1
      x1 <- rexp(j) - 1
      x2 <- runif(j, -sqrt(3), sqrt(3))
      x3 <- rnorm(j)
      X <- cbind(x1, x2, x3)
      MD.fobi[row] <- MD(coef(FOBI(X)), A)
      MD.jade[row] <- MD(coef(JADE(X)), A)
    }
  }
  RES <- data.frame(N = rep(ns, each = repet), fobi = MD.fobi, jade = MD.jade)
  RES
}

set.seed(123)
N <- 2^((-2):5) * 1000
MDs <- ICAsim(ns = N, repet = 2000)

f1 <- function(x) { exp(-x - 1) }
f2 <- function(x) { rep(1 / (2 * sqrt(3)), length(x)) }
f3 <- function(x) { exp(-(x)^2 / 2) / sqrt(2 * pi) }
support <- matrix(c(-1, -sqrt(3), -Inf, Inf, sqrt(3), Inf), nrow = 3)
fobi <- ASCOV_FOBI(sdf = c(f1, f2, f3), supp = support)
jade <- ASCOV_JADE(sdf = c(f1, f2, f3), supp = support)

fobi$W

fobi$COV_W

fobi$EMD

jade$EMD

meanMDs <- aggregate(MDs[, 2:3]^2, list(N = MDs$N), mean)
MmeansMDs <- 2 * meanMDs[, 1] * meanMDs[, 2:3]
ylabel <- expression(paste("n(p-1)ave", (hat(D)^2)))
par(mar = c(4, 5, 0, 0) + 0.1)
matplot(N, MmeansMDs, pch = c(15, 16), ylim = c(0, 60), ylab = ylabel, log = "x", xlab = "n", cex = c(1.5, 1.2), col = c(1, 4), xaxt = "n")
axis(1, N)
abline(h = fobi$EMD, lty = 1, lwd = 2)
abline(h = jade$EMD, lty = 2, col = 4, lwd = 2)
legend("topright", c("FOBI", "JADE"), lty = c(1, 2), pch = 15:17, col = c(1, 4), bty = "n", pt.cex = c(1.5, 1.2), lwd = 2)
