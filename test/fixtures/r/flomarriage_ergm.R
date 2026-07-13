# Golden fixture: statnet `ergm` FITTED OUTPUT on the Florentine marriage network.
#
# Regenerate from the package root (~60s: the replication study below refits the
# dyad-dependent model under five further seeds):
#
#   Rscript test/fixtures/r/flomarriage_ergm.R > test/fixtures/flomarriage_ergm.toml
#
# WHY TWO MODELS, AND WHY THAT IS THE WHOLE POINT
#
# ERGM.jl's test suite used to carry R's Florentine coefficients as bare numbers
# in comments with hand-picked atols. That checks something, but it cannot tell
# you WHICH of the two very different things it is checking, because an ERGM fit
# is two different kinds of number depending on the model:
#
#   (a) DYAD-INDEPENDENT (edges + nodecov("wealth")). The likelihood factorizes
#       over dyads, so the pseudo-likelihood IS the likelihood: MPLE == exact ML.
#       Both implementations are solving the same convex logistic regression to
#       convergence. There is no Monte Carlo anywhere. Agreement should be at
#       optimizer precision (~1e-6), and anything worse is a BUG, not noise.
#       This is the half that can be tested exactly, and it is tested exactly.
#
#   (b) DYAD-DEPENDENT (edges + gwesp(0.5, fixed=TRUE)). No factorization, no
#       closed-form likelihood; both implementations run MCMC-MLE, and each run
#       lands somewhere in a cloud around the MLE. Machine precision here would
#       be a category error -- but "it's Monte Carlo" is not a licence to accept
#       anything either. So this script MEASURES the cloud: it refits the
#       identical model under five further ergm seeds and emits the seed-to-seed
#       sd of every coefficient (`mcmle_seed_sd`). That is R disagreeing with
#       ITSELF, and it is the floor below which no cross-implementation tolerance
#       can honestly be set. The tolerances in [tolerance] are stated as
#       multiples of it, and of the fitted standard errors -- the scale at which
#       a difference would actually change a published conclusion.
#
# Summary statistics are frozen too, at 1e-9: they are a deterministic function
# of the observed graph, so any disagreement is a bug in a term formula, full
# stop, with no Monte Carlo to hide behind.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm)
})

seed <- 20260713
set.seed(seed)

data(florentine)
flo <- flomarriage

# `ergm()` prints its MCMLE iteration chatter to stdout, and this script's stdout
# IS the TOML fixture, so unsuppressed it would emit an unparseable file. Capture
# and discard; `verbose=FALSE` is not enough on its own.
quiet_ergm <- function(...) {
  fit <- NULL
  invisible(capture.output(fit <- ergm(...), type = "output"))
  fit
}

# --- (a) dyad-independent: MPLE is the exact MLE ----------------------------
# ergm() detects dyad-independence and fits by (penalized-free) logistic
# regression rather than MCMC; `estimate="MLE"` is the default and lands on the
# same numbers as estimate="MPLE" here, by construction.
f_di <- flo ~ edges + nodecov("wealth")
fit_di <- quiet_ergm(f_di)
sum_di <- summary(f_di)

# --- (b) dyad-dependent: MCMLE --------------------------------------------
f_dd <- flo ~ edges + gwesp(0.5, fixed = TRUE)
sum_dd <- summary(f_dd)

fit_dd_once <- function(s) {
  set.seed(s)
  quiet_ergm(f_dd, control = control.ergm(seed = s, MCMC.samplesize = 4096,
                                          MCMC.burnin = 16384,
                                          MCMC.interval = 1024))
}
fit_dd <- fit_dd_once(seed)

# How much does ergm disagree with ITSELF? Five further seeds, same data, same
# model, same MCMC budget. Pure Monte-Carlo width of the MCMLE estimator.
rep_seeds <- c(101, 202, 303, 404, 505)
reps <- t(sapply(rep_seeds, function(s) coef(fit_dd_once(s))))
seed_sd <- apply(reps, 2, sd)

se_of <- function(fit) sqrt(diag(vcov(fit)))
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")

cat('name = "flomarriage_ergm"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/flomarriage_ergm.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "ergm::flomarriage (Padgett): 16 Florentine families, 20 undirected marriage ties, wealth covariate"\n')
cat('model_dyad_independent = "flomarriage ~ edges + nodecov(\\"wealth\\") -- fitted by ergm(), which uses logistic regression (MPLE = exact MLE) for dyad-independent formulas"\n')
cat('model_dyad_dependent = "flomarriage ~ edges + gwesp(0.5, fixed=TRUE) -- fitted by MCMLE"\n')
cat('mcmc_control = "control.ergm(MCMC.samplesize=4096, MCMC.burnin=16384, MCMC.interval=1024)"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# Summary statistics are a DETERMINISTIC function of the observed graph --\n")
cat("# no estimator, no simulation. Machine precision, and any disagreement is a\n")
cat("# bug in a term formula, full stop.\n")
cat("summary_statistics = 1e-9\n")
cat("#\n")
cat("# DYAD-INDEPENDENT FIT. The likelihood factorizes over dyads, so the\n")
cat("# pseudo-likelihood IS the likelihood and both implementations are solving\n")
cat("# the SAME strictly-convex logistic regression to convergence. No Monte\n")
cat("# Carlo is involved on either side. The only difference two correct\n")
cat("# implementations can have is optimizer termination: R uses IRLS to ~1e-8,\n")
cat("# ERGM.jl uses L-BFGS. 1e-6 is well inside both, and is ~1e-4 of the\n")
cat("# smallest standard error in the model. If this fails it is a BUG -- do not\n")
cat("# loosen it.\n")
cat("di_coefficients = 1e-6\n")
cat("di_std_errors = 1e-6\n")
cat("#\n")
cat("# DYAD-DEPENDENT FIT (gwesp). Read `mcmle_seed_sd` in [values] first: that\n")
cat("# is R refitting this very model under five further seeds and disagreeing\n")
cat("# with itself. It is the Monte-Carlo width of the MCMLE estimator, and no\n")
cat("# cross-implementation tolerance can honestly sit below it.\n")
cat("#\n")
cat("# Measured: R's seed-to-seed sd here is 0.0057 (edges) and 0.0059 (gwesp),\n")
cat("# and the frozen fit at seed 20260713 sits ~1.1 of those sds from R's own\n")
cat("# five-seed mean -- so the fixture VALUE itself carries that much noise, and\n")
cat("# the tolerance has to cover R's single-run error before it covers anything\n")
cat("# about Julia.\n")
cat("#\n")
cat("# The Julia side compares the MEAN of five ERGM.jl MCMLE fits at declared\n")
cat("# seeds, whose own Monte-Carlo error is sd/sqrt(5). Observed gaps:\n")
cat("#   |mean(ERGM.jl) - R| coefficients: 0.0016 (edges), 0.00034 (gwesp)\n")
cat("#   |mean(ERGM.jl) - R| std errors  : 0.0052 (edges), 0.00075 (gwesp)\n")
cat("# Every one of those is BELOW R's own seed-to-seed sd -- the two\n")
cat("# implementations differ by less than R differs from itself.\n")
cat("#\n")
cat("# 0.03 is ~5x R's single-run sd (so a green test is not luck) and is 8% of\n")
cat("# the edges standard error and 11% of the gwesp standard error -- a\n")
cat("# discrepancy large enough to move a published conclusion cannot hide under\n")
cat("# it. It is NOT chosen to make the test pass: it has a ~6x margin over the\n")
cat("# largest gap actually observed.\n")
cat("dd_coefficients = 0.03\n")
cat("dd_std_errors = 0.03\n")
cat("#\n")
cat("# NOTE, on the record: ERGM.jl's MCMLE point estimate for this model has\n")
cat("# ZERO seed-to-seed variance, because its convergence test fires at\n")
cat("# iteration 1 (max t-ratio 0.006, Hotelling p = 0.87) BEFORE any Newton\n")
cat("# update is applied, so it returns the MPLE unchanged. The MPLE happens to\n")
cat("# satisfy the moment condition E_theta[g] = g_obs to within Monte-Carlo\n")
cat("# error on this network, so that is a defensible stopping point rather than\n")
cat("# a wrong answer -- and indeed it lands within R's own noise. But statnet\n")
cat("# always takes at least one MCMLE step, and a reader comparing the two\n")
cat("# should know the Julia number is a pseudo-likelihood estimate that passed\n")
cat("# an MCMC convergence check, not the output of an MCMC-driven update. The\n")
cat("# ERGM.jl testset asserts this explicitly so the day it changes is visible.\n\n")

cat("[values]\n")
cat("# --- observed graph, deterministic ---------------------------------------\n")
cat(sprintf("summary_statistic_names = [%s]\n",
            strs(c(names(sum_di), setdiff(names(sum_dd), names(sum_di))))))
cat(sprintf("summary_statistics = [%s]\n",
            num(c(sum_di, sum_dd[setdiff(names(sum_dd), names(sum_di))]))))
cat("\n# --- (a) dyad-independent: MPLE == exact ML, compared at 1e-6 ------------\n")
cat(sprintf("di_terms = [%s]\n", strs(names(coef(fit_di)))))
cat(sprintf("di_coefficients = [%s]\n", num(coef(fit_di))))
cat(sprintf("di_std_errors = [%s]\n", num(se_of(fit_di))))
cat(sprintf("di_loglik = %.17g\n", as.numeric(logLik(fit_di))))
cat(sprintf("di_aic = %.17g\n", AIC(fit_di)))
cat("\n# --- (b) dyad-dependent: MCMLE, compared against measured MC width -------\n")
cat(sprintf("dd_terms = [%s]\n", strs(names(coef(fit_dd)))))
cat(sprintf("dd_coefficients = [%s]\n", num(coef(fit_dd))))
cat(sprintf("dd_std_errors = [%s]\n", num(se_of(fit_dd))))
cat("\n# ergm disagreeing with ITSELF across five further seeds: the Monte-Carlo\n")
cat("# floor under every dyad-dependent tolerance above.\n")
cat(sprintf("mcmle_seed_sd = [%s]\n", num(seed_sd)))
cat(sprintf("mcmle_seed_mean = [%s]\n", num(colMeans(reps))))
cat(sprintf("mcmle_seed_min = [%s]\n", num(apply(reps, 2, min))))
cat(sprintf("mcmle_seed_max = [%s]\n", num(apply(reps, 2, max))))
