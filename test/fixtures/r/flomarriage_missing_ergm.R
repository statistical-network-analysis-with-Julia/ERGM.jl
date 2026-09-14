# Golden fixture: statnet `ergm` on the Florentine marriage network with FOUR
# DYADS MASKED AS NA (unobserved) -- the R reference for ERGM.jl's missing-data
# treatments (panel 2026-09, item 32 / N5).
#
# Regenerate from the package root (~2 min: the replication study below refits
# the dyad-dependent missing-data MLE under five further seeds):
#
#   Rscript test/fixtures/r/flomarriage_missing_ergm.R > test/fixtures/flomarriage_missing_ergm.toml
#
# WHAT IS MASKED
#
# Exactly four dyads are set to NA: (3,4) and (2,11), which are NOT ties in
# flomarriage, and (1,9) and (7,16), which ARE ties. Two unobserved non-ties
# and two unobserved ties, so a treatment that silently reads the face value
# of a masked dyad is caught in both directions. The network is otherwise
# `ergm::flomarriage` unchanged; `[provenance] masked_dyads` lists them.
#
# WHAT R DOES WITH NA DYADS, AND WHAT IS PINNED
#
#   (a) summary(): statnet evaluates the statistics on the network with every
#       NA dyad treated as ABSENT (network.edgecount(na.omit=TRUE)): 18 edges,
#       not 20. Deterministic; frozen at 1e-9. ERGM.jl's `compute` reads the
#       stored face value, so the Julia side must evaluate the statistics on a
#       copy with the masked ties removed to reproduce these numbers -- the
#       testset says so.
#
#   (b) DYAD-INDEPENDENT (edges + nodecov("wealth")): with NA dyads ergm()'s
#       MLE is the logistic regression on the 116 OBSERVED dyads (nobs = 116),
#       which is exactly ERGM.jl's available-case MPLE. Same convex problem on
#       both sides, no Monte Carlo: agreement at optimizer precision, and a
#       failure is a BUG, not noise. Frozen at 1e-6.
#
#   (c) DYAD-DEPENDENT (edges + gwesp(0.5, fixed=TRUE)): ergm() runs the
#       missing-data MCMLE of Handcock & Gile (2010) -- a constrained chain
#       that toggles only the NA dyads estimates E[g(Y) | Y_obs], the free
#       chain estimates E[g(Y)], and the estimate solves their equality. Both
#       implementations are Monte Carlo, so this script MEASURES the width:
#       it refits under five further seeds and emits R's own seed-to-seed sd
#       (`mcmle_seed_sd`), the floor under any honest cross-implementation
#       tolerance. R's "MCMC %" column is recorded too.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm)
})

seed <- 20260909
set.seed(seed)

data(florentine)
flo <- flomarriage
masked <- list(c(3, 4), c(1, 9), c(7, 16), c(2, 11))
masked_face <- sapply(masked, function(d) flo[d[1], d[2]])   # 0, 1, 1, 0
for (d in masked) flo[d[1], d[2]] <- NA
stopifnot(network.naedgecount(flo) == 4)

# `ergm()` prints its MCMLE iteration chatter to stdout, and this script's stdout
# IS the TOML fixture, so unsuppressed it would emit an unparseable file. Capture
# and discard; `verbose=FALSE` is not enough on its own.
quiet_ergm <- function(...) {
  fit <- NULL
  invisible(capture.output(fit <- ergm(...), type = "output"))
  fit
}

# --- (a) summary statistics on the masked network --------------------------
f_sum <- flo ~ edges + nodecov("wealth") + gwesp(0.5, fixed = TRUE)
sum_masked <- summary(f_sum)

# --- (b) dyad-independent: logistic regression on the observed dyads --------
f_di <- flo ~ edges + nodecov("wealth")
fit_di <- quiet_ergm(f_di)

# --- (c) dyad-dependent: missing-data MCMLE ---------------------------------
f_dd <- flo ~ edges + gwesp(0.5, fixed = TRUE)

fit_dd_once <- function(s) {
  set.seed(s)
  quiet_ergm(f_dd, control = control.ergm(seed = s, MCMC.samplesize = 4096,
                                          MCMC.burnin = 16384,
                                          MCMC.interval = 1024))
}
fit_dd <- fit_dd_once(seed)
mcmc_pct <- summary(fit_dd)$coefficients[, "MCMC %"]

# How much does ergm disagree with ITSELF? Five further seeds, same data, same
# model, same MCMC budget. Pure Monte-Carlo width of the missing-data MCMLE.
rep_seeds <- c(101, 202, 303, 404, 505)
rep_fits <- lapply(rep_seeds, fit_dd_once)
reps <- t(sapply(rep_fits, coef))
rep_ses <- t(sapply(rep_fits, function(f) sqrt(diag(vcov(f)))))
seed_sd <- apply(reps, 2, sd)
seed_sd_se <- apply(rep_ses, 2, sd)

se_of <- function(fit) sqrt(diag(vcov(fit)))
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
fmt3 <- function(x) paste(sprintf("%.3g", x), collapse = ", ")

# The dyad-dependent tolerance is stated as a multiple of the MEASURED
# seed-to-seed sd. It is a fixed number so that the Julia assertion cannot
# drift with a lucky regeneration, and the script REFUSES to emit a fixture
# whose measured width would not justify it.
tol_dd <- 0.03
if (5 * max(seed_sd) > tol_dd || 3 * max(seed_sd_se) > tol_dd) {
  stop(sprintf("measured seed sd (%s / SE sd %s) no longer justifies tol_dd = %g",
               num(seed_sd), num(seed_sd_se), tol_dd))
}

cat('name = "flomarriage_missing_ergm"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/flomarriage_missing_ergm.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "ergm::flomarriage (Padgett): 16 Florentine families, 20 undirected marriage ties, wealth covariate; four dyads set to NA"\n')
cat(sprintf('masked_dyads = "%s"\n',
            paste(sapply(masked, function(d) sprintf("(%d,%d)", d[1], d[2])), collapse = ", ")))
cat(sprintf('masked_face_values = "%s -- 1 = the masked dyad is a tie in flomarriage, 0 = it is not"\n',
            paste(masked_face, collapse = ", ")))
cat('summary_convention = "statnet summary() evaluates the statistics with every NA dyad treated as absent (network.edgecount(na.omit=TRUE))"\n')
cat('model_dyad_independent = "flomarriage[NA] ~ edges + nodecov(\\"wealth\\") -- ergm() fits the logistic regression on the 116 observed dyads (= available-case MPLE = exact MLE under dyad independence)"\n')
cat('model_dyad_dependent = "flomarriage[NA] ~ edges + gwesp(0.5, fixed=TRUE) -- missing-data MCMLE (Handcock & Gile 2010): constrained chain over the NA dyads"\n')
cat('mcmc_control = "control.ergm(MCMC.samplesize=4096, MCMC.burnin=16384, MCMC.interval=1024)"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# Summary statistics on the masked network are a DETERMINISTIC function of\n")
cat("# the observed graph under statnet's NA-as-absent convention. Machine\n")
cat("# precision; any disagreement is a bug in a term formula or in the masking.\n")
cat("summary_statistics = 1e-9\n")
cat("#\n")
cat("# DYAD-INDEPENDENT FIT WITH NA DYADS. ergm() drops the NA dyads and fits the\n")
cat("# logistic regression on the 116 observed ones -- the SAME strictly-convex\n")
cat("# problem ERGM.jl's available-case MPLE solves. No Monte Carlo on either\n")
cat("# side; the only difference is optimizer termination (R IRLS ~1e-8, ERGM.jl\n")
cat("# Newton-Raphson 1e-8). 1e-6 is well inside both. If this fails it is a\n")
cat("# BUG -- do not loosen it.\n")
cat("di_coefficients = 1e-6\n")
cat("di_std_errors = 1e-6\n")
cat("#\n")
cat("# DYAD-DEPENDENT MISSING-DATA MCMLE. Read `mcmle_seed_sd` in [values]: R\n")
cat("# refitting this model under five further seeds and disagreeing with itself.\n")
cat("# No cross-implementation tolerance can honestly sit below that width.\n")
cat(sprintf("# Measured: seed-to-seed sd of the coefficients %s;\n", fmt3(seed_sd)))
cat(sprintf("# of the standard errors %s. The frozen fit at\n", fmt3(seed_sd_se)))
cat(sprintf("# seed %d sits %s sds (edges, gwesp) from R's own five-seed mean.\n", seed,
            fmt3(abs(coef(fit_dd) - colMeans(reps)) / seed_sd)))
cat("#\n")
cat("# The Julia side compares the MEAN of five ERGM.jl `missing=:mle` fits at\n")
cat("# declared seeds (MC error sd/sqrt(5)) against the frozen R fit.\n")
cat(sprintf("# %g is >= 5x the largest measured coefficient seed sd (%s) -- the script\n",
            tol_dd, fmt3(5 * max(seed_sd))))
cat("# refuses to emit the fixture otherwise -- and it is a small fraction of the\n")
cat(sprintf("# fitted standard errors (%s), so a discrepancy large enough to\n",
            fmt3(se_of(fit_dd))))
cat("# move a published conclusion cannot hide under it.\n")
cat(sprintf("dd_coefficients = %g\n", tol_dd))
cat(sprintf("dd_std_errors = %g\n", tol_dd))
cat("\n")

cat("[values]\n")
cat("# --- (a) observed graph with NA dyads absent, deterministic --------------\n")
cat(sprintf("summary_statistic_names = [%s]\n", strs(names(sum_masked))))
cat(sprintf("summary_statistics = [%s]\n", num(sum_masked)))
cat(sprintf("n_observed_dyads = %d\n", as.integer(nobs(fit_di))))
cat("\n# --- (b) dyad-independent: available-case MLE, compared at 1e-6 ---------\n")
cat(sprintf("di_terms = [%s]\n", strs(names(coef(fit_di)))))
cat(sprintf("di_coefficients = [%s]\n", num(coef(fit_di))))
cat(sprintf("di_std_errors = [%s]\n", num(se_of(fit_di))))
cat(sprintf("di_loglik = %.17g\n", as.numeric(logLik(fit_di))))
cat(sprintf("di_aic = %.17g\n", AIC(fit_di)))
cat("\n# --- (c) dyad-dependent: missing-data MCMLE, vs measured MC width --------\n")
cat(sprintf("dd_terms = [%s]\n", strs(names(coef(fit_dd)))))
cat(sprintf("dd_coefficients = [%s]\n", num(coef(fit_dd))))
cat(sprintf("dd_std_errors = [%s]\n", num(se_of(fit_dd))))
cat(sprintf("dd_mcmc_percent = [%s]\n", num(mcmc_pct)))
cat("\n# ergm disagreeing with ITSELF across five further seeds: the Monte-Carlo\n")
cat("# floor under every dyad-dependent tolerance above.\n")
cat(sprintf("mcmle_seed_sd = [%s]\n", num(seed_sd)))
cat(sprintf("mcmle_seed_sd_std_errors = [%s]\n", num(seed_sd_se)))
cat(sprintf("mcmle_seed_mean = [%s]\n", num(colMeans(reps))))
cat(sprintf("mcmle_seed_min = [%s]\n", num(apply(reps, 2, min))))
cat(sprintf("mcmle_seed_max = [%s]\n", num(apply(reps, 2, max))))
