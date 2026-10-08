# Golden fixture: statnet `ergm` CURVED exponential-family fit -- the decay of
# a geometrically weighted term estimated (gwesp(decay, fixed=FALSE)).
#
# Regenerate from the package root (~5 min):
#
#   Rscript test/fixtures/r/curved_ergm.R > test/fixtures/curved_ergm.toml
#
# The model is the Goodreau et al. (2008) faux.mesa.high model with the
# gwesp decay free:
#   faux.mesa.high ~ edges + nodematch("Grade") + nodematch("Race") +
#                    gwesp(0.25, fixed=FALSE)
# fitted by MCMLE at ergm()'s defaults from the start (-6.41, 1.99, 0.29,
# 1.45, 0.25) -- the fixed-decay estimates and decay 0.25 -- under ten seeds.
# R's curved MCMLE does not finish under every seed (an excursion past the
# term's shared-partner cut-off stops some runs: "number of ... two-paths on
# some edge exceeded the cut-off"); the seeds that finish are the reference,
# and their number is recorded. The comparison is R's seed-to-seed spread, as
# in mcmle_ergm.toml: an MCMLE is a Monte-Carlo estimate.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm)
})
seed <- 20261002
data(faux.mesa.high)
f <- faux.mesa.high ~ edges + nodematch("Grade") + nodematch("Race") + gwesp(0.25, fixed = FALSE)
init <- c(-6.41, 1.99, 0.29, 1.45, 0.25)
quiet <- function(expr) {
  out <- NULL
  invisible(capture.output(out <- suppressMessages(suppressWarnings(expr)), type = "output"))
  out
}
seeds <- c(seed, 101, 202, 303, 404, 505, 606, 707, 808, 909)
fits <- lapply(seeds, function(s) {
  set.seed(s)
  tryCatch(quiet(ergm(f, control = control.ergm(seed = s, init = init))), error = function(e) NULL)
})
ok <- !sapply(fits, is.null)
fits <- fits[ok]
stopifnot(length(fits) >= 5)
se_of <- function(fit) sqrt(diag(vcov(fit)))
cm <- t(sapply(fits, coef)); sm <- t(sapply(fits, se_of))
ll <- sapply(fits, function(x) as.numeric(logLik(x)))
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
k <- length(fits)
tol4 <- function(sd, se) pmax(4 * sqrt(sd^2 + sd^2 / k), 0.1 * se)

cat('name = "curved_ergm"\n\n[provenance]\n')
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/curved_ergm.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('model = "faux.mesa.high ~ edges + nodematch(\\"Grade\\") + nodematch(\\"Race\\") + gwesp(0.25, fixed=FALSE); control.ergm(seed=s, init=c(-6.41, 1.99, 0.29, 1.45, 0.25))"\n')
cat(sprintf('seeds_tried = "%s"\n', paste(seeds, collapse = ",")))
cat(sprintf('seeds_finished = "%s"\n', paste(seeds[ok], collapse = ",")))
cat("\n[tolerance]\n")
cat("# One ERGM.jl curved MCMLE fit against the mean of R's finished fits, per\n")
cat("# coefficient: 4*sqrt(sd_R^2 + sd_R^2/k) floored at a tenth of R's standard\n")
cat("# error (the rule of mcmle_ergm.toml), emitted as coef_tolerance; standard\n")
cat("# errors by the same rule on R's SE seed sd, doubled (a curved fit's\n")
cat("# information matrix is estimated from the final sample alone). The\n")
cat("# log-likelihood: within 4 of R's seed sds of R's mean (two bridges).\n")
cat("loglik_sd_multiple = 4\n\n[values]\n")
cat(sprintf("terms = [%s]\n", strs(colnames(cm))))
cat(sprintf("n_finished = %d\n", k))
cat(sprintf("coef_mean = [%s]\n", num(colMeans(cm))))
cat(sprintf("coef_seed_sd = [%s]\n", num(apply(cm, 2, sd))))
cat(sprintf("se_mean = [%s]\n", num(colMeans(sm))))
cat(sprintf("se_seed_sd = [%s]\n", num(apply(sm, 2, sd))))
cat(sprintf("coef_tolerance = [%s]\n", num(tol4(apply(cm, 2, sd), colMeans(sm)))))
cat(sprintf("se_tolerance = [%s]\n", num(2 * tol4(apply(sm, 2, sd), colMeans(sm)))))
cat(sprintf("loglik_mean = %.17g\n", mean(ll)))
cat(sprintf("loglik_seed_sd = %.17g\n", sd(ll)))
