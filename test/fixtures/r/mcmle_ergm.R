# Golden fixture: statnet `ergm` MCMLE on the canonical dyad-dependent teaching
# models -- the reference for ERGM.jl's MCMLE algorithm (tie/no-tie proposal,
# R's `confidence` stopping rule, at least one Monte-Carlo step) and for its
# bridge log-likelihood.
#
# Regenerate from the package root (~8 min: every model is refitted under ten
# further seeds, and the bridge is rerun under ten seeds):
#
#   Rscript test/fixtures/r/mcmle_ergm.R > test/fixtures/mcmle_ergm.toml
#
# WHAT IS FROZEN, AND HOW IT MAY BE COMPARED
#
# (a) faux.mesa.high ~ edges + nodematch("Grade") + nodematch("Race") +
#     gwesp(0.25, fixed=TRUE) -- the Goodreau et al. (2008) / statnet tutorial
#     model -- and (b) the same plus gwdegree(0.5, fixed=TRUE), each fitted by
#     ergm() at ITS DEFAULTS (control.ergm(seed=s) only), under eleven seeds. An
#     MCMLE is a Monte-Carlo estimate: two correct implementations agree only
#     up to the estimator's Monte-Carlo width, which this script MEASURES as
#     R's seed-to-seed sd of every coefficient and standard error. The
#     tolerances below are multiples of that width (see [tolerance]); none is
#     chosen to make a test pass.
#
# (c) flomarriage ~ edges + triangle: the high-precision MLE (50 000 draws,
#     R's `precision` termination), the reference against which a mean of
#     ERGM.jl MCMLE fits is checked -- the model on which ERGM.jl's pre-0.2
#     MCMLE returned the MPLE unchanged in 12 of 30 seeds.
#
# (d) The bridge log-likelihood of model (a) at R's frozen coefficients,
#     estimated by R's `ergm.bridge.dindstart.llk` with 16 and with 64 bridge
#     steps under five seeds each: the 64-step mean is the reference, the
#     16-step seed sd the Monte-Carlo width of one estimate (ten seeds each).
#
# Summary statistics are frozen at 1e-9: a deterministic function of the
# observed graph, they also prove the two packages fit the same data.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm)
})

seed <- 20261002
set.seed(seed)
data(faux.mesa.high)
data(florentine)
fmh <- faux.mesa.high

quiet <- function(expr) {
  out <- NULL
  invisible(capture.output(out <- suppressMessages(suppressWarnings(expr)), type = "output"))
  out
}

f_a <- fmh ~ edges + nodematch("Grade") + nodematch("Race") + gwesp(0.25, fixed = TRUE)
f_b <- fmh ~ edges + nodematch("Grade") + nodematch("Race") + gwesp(0.25, fixed = TRUE) +
  gwdegree(0.5, fixed = TRUE)
sum_a <- summary(f_a)
sum_b <- summary(f_b)

fit_at <- function(f, s) {
  set.seed(s)
  quiet(ergm(f, control = control.ergm(seed = s)))
}
se_of <- function(fit) sqrt(diag(vcov(fit)))
rep_seeds <- c(101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)

study <- function(f) {
  fit <- fit_at(f, seed)
  reps <- lapply(rep_seeds, function(s) fit_at(f, s))
  cm <- t(sapply(reps, coef))
  sm <- t(sapply(reps, se_of))
  list(fit = fit, coef_mean = colMeans(rbind(cm, coef(fit))),
       coef_sd = apply(rbind(cm, coef(fit)), 2, sd),
       se_mean = colMeans(rbind(sm, se_of(fit))),
       se_sd = apply(rbind(sm, se_of(fit)), 2, sd),
       loglik = c(sapply(reps, function(r) as.numeric(logLik(r))), as.numeric(logLik(fit))))
}
A <- study(f_a)
B <- study(f_b)

# (c) Florentine edges + triangle at high precision
set.seed(11)
fit_tri <- quiet(ergm(flomarriage ~ edges + triangle,
                      control = control.ergm(seed = 11, MCMC.samplesize = 50000,
                                             MCMC.interval = 2048,
                                             MCMLE.termination = "precision",
                                             MCMLE.maxit = 30)))
mple_tri <- quiet(ergm(flomarriage ~ edges + triangle, estimate = "MPLE"))
tri_reps <- t(sapply(1:10, function(s) {
  set.seed(s); coef(quiet(ergm(flomarriage ~ edges + triangle, control = control.ergm(seed = s))))
}))

# (d) bridge log-likelihood of (a) at the frozen coefficients
th_a <- coef(A$fit)
bridge <- function(ns) sapply(1:10, function(s) {
  set.seed(s)
  quiet(ergm.bridge.dindstart.llk(f_a, coef = th_a, llkonly = TRUE,
                                  control = control.ergm.bridge(bridge.nsteps = ns)))
})
llk16 <- bridge(16)
llk64 <- bridge(64)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
n_fits <- length(rep_seeds) + 1
tol4 <- function(sd, se) pmax(4 * sqrt(sd^2 + sd^2 / n_fits), 0.1 * se)   # see [tolerance]

cat('name = "mcmle_ergm"\n\n')
cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/mcmle_ergm.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "ergm::faux.mesa.high (205 students, 203 undirected ties, Grade/Race/Sex) and ergm::flomarriage"\n')
cat('model_a = "faux.mesa.high ~ edges + nodematch(\\"Grade\\") + nodematch(\\"Race\\") + gwesp(0.25, fixed=TRUE)"\n')
cat('model_b = "model_a + gwdegree(0.5, fixed=TRUE)"\n')
cat('model_c = "flomarriage ~ edges + triangle"\n')
cat('control = "ergm() defaults: control.ergm(seed=s); (c) high precision: MCMC.samplesize=50000, MCMC.interval=2048, MCMLE.termination=\\"precision\\""\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# Deterministic functions of the observed graph: machine precision.\n")
cat("a_summary = 1e-9\n")
cat("b_summary = 1e-9\n")
cat("# The Florentine edges + triangle MPLE: the same convex logistic regression\n")
cat("# solved to convergence on both sides -- optimizer precision.\n")
cat("tri_mple = 1e-6\n")
cat("#\n")
cat("# (a)/(b) A SINGLE ERGM.jl MCMLE fit at its defaults is compared with R's\n")
cat("# eleven-seed mean. The difference has variance sd_J^2 + sd_R^2/11; ERGM.jl's\n")
cat("# own seed-to-seed sd (measured 0.0013-0.0061 on these models, 5 seeds) is\n")
cat("# of R's order, so sd_R stands in for both: the per-coefficient tolerance is\n")
cat("# 4*sqrt(sd_R^2 + sd_R^2/11) -- four standard deviations of the difference\n")
cat("# under the hypothesis that the two estimators agree -- floored at a tenth\n")
cat("# of R's standard error, the practical-equivalence margin of the flomarriage\n")
cat("# fixture (0.03 there, 8-11% of a standard error). The floor is needed: two\n")
cat("# MCMLE implementations with different MCMC designs (proposal, sample-size\n")
cat("# adaptation) carry different O(1/ESS) finite-sample biases, and R's seed sd\n")
cat("# on gwesp here (0.0015 over eleven fits) is below that. Measured before\n")
cat("# freezing: five default ERGM.jl fits of (a) sit 0.0016-0.0076 from R's mean\n")
cat("# on gwesp and 0.0014-0.010 on the other coefficients -- at most 7% of a\n")
cat("# standard error. Emitted per coefficient in [values] as a_coef_tolerance /\n")
cat("# b_coef_tolerance; the standard errors use the same rule on R's SE seed sd.\n")
cat("#\n")
cat("# (c) No ERGM.jl MCMLE fit may return tri_mple verbatim (the pre-0.2\n")
cat("# defect: the convergence test ran before the first step), and the mean of\n")
cat("# ten ERGM.jl fits must lie within 4 standard errors of the mean -- R's\n")
cat("# ten-seed sd over sqrt(10), the width of R's own default estimator -- of\n")
cat("# the high-precision MLE.\n")
cat("tri_mean_sd_multiple = 4\n")
cat("#\n")
cat("# (d) One ERGM.jl bridge estimate (16 Simpson rungs) at R's coefficients\n")
cat("# against R's 64-step mean: 4 x R's 16-step seed sd (the width of one\n")
cat("# default estimate). The quadrature bias of the pre-0.2 trapezoid (-0.3 to\n")
cat("# -0.5 here) is below that width; it is pinned exactly, on an enumerable\n")
cat("# network, by the ERGM.jl testset instead.\n")
cat("bridge_sd_multiple = 4\n\n")

cat("[values]\n")
cat(sprintf("a_terms = [%s]\n", strs(names(sum_a))))
cat(sprintf("a_summary = [%s]\n", num(sum_a)))
cat(sprintf("b_terms = [%s]\n", strs(names(sum_b))))
cat(sprintf("b_summary = [%s]\n", num(sum_b)))
cat("\n# --- (a) gwesp model at ergm() defaults, eleven seeds -------------------------\n")
cat(sprintf("a_coefficients = [%s]\n", num(coef(A$fit))))
cat(sprintf("a_std_errors = [%s]\n", num(se_of(A$fit))))
cat(sprintf("a_coef_mean = [%s]\n", num(A$coef_mean)))
cat(sprintf("a_coef_seed_sd = [%s]\n", num(A$coef_sd)))
cat(sprintf("a_se_mean = [%s]\n", num(A$se_mean)))
cat(sprintf("a_se_seed_sd = [%s]\n", num(A$se_sd)))
cat(sprintf("a_coef_tolerance = [%s]\n", num(tol4(A$coef_sd, A$se_mean))))
cat(sprintf("a_se_tolerance = [%s]\n", num(tol4(A$se_sd, A$se_mean))))
cat(sprintf("a_loglik = [%s]\n", num(A$loglik)))
cat("\n# --- (b) + gwdegree(0.5), eleven seeds -----------------------------------------\n")
cat(sprintf("b_coefficients = [%s]\n", num(coef(B$fit))))
cat(sprintf("b_std_errors = [%s]\n", num(se_of(B$fit))))
cat(sprintf("b_coef_mean = [%s]\n", num(B$coef_mean)))
cat(sprintf("b_coef_seed_sd = [%s]\n", num(B$coef_sd)))
cat(sprintf("b_se_mean = [%s]\n", num(B$se_mean)))
cat(sprintf("b_se_seed_sd = [%s]\n", num(B$se_sd)))
cat(sprintf("b_coef_tolerance = [%s]\n", num(tol4(B$coef_sd, B$se_mean))))
cat(sprintf("b_se_tolerance = [%s]\n", num(tol4(B$se_sd, B$se_mean))))
cat(sprintf("b_loglik = [%s]\n", num(B$loglik)))
cat("\n# --- (c) flomarriage edges + triangle -----------------------------------------\n")
cat(sprintf("tri_mple = [%s]\n", num(coef(mple_tri))))
cat(sprintf("tri_mle_precise = [%s]\n", num(coef(fit_tri))))
cat(sprintf("tri_mle_precise_se = [%s]\n", num(se_of(fit_tri))))
cat(sprintf("tri_seed_mean = [%s]\n", num(colMeans(tri_reps))))
cat(sprintf("tri_seed_sd = [%s]\n", num(apply(tri_reps, 2, sd))))
cat(sprintf("tri_seed_min = [%s]\n", num(apply(tri_reps, 2, min))))
cat(sprintf("tri_seed_max = [%s]\n", num(apply(tri_reps, 2, max))))
cat("\n# --- (d) bridge log-likelihood of (a) at a_coefficients ----------------------\n")
cat(sprintf("bridge16 = [%s]\n", num(llk16)))
cat(sprintf("bridge64 = [%s]\n", num(llk64)))
cat(sprintf("bridge64_mean = %.17g\n", mean(llk64)))
cat(sprintf("bridge16_seed_sd = %.17g\n", sd(llk16)))
