# Golden fixture: statnet `ergm` on statistics at the boundary of their
# attainable range (R's `ergm.checkextreme.model` and its default
# `drop=TRUE`) and on statistics without an identifiable coefficient.
#
# Regenerate from the package root (~8 min):
#
#   Rscript test/fixtures/r/boundary_ergm.R > test/fixtures/boundary_ergm.toml
#
# (a) A STATISTIC WHOSE CHANGE STATISTICS ARE ALL ZERO: a 10-node perfect
#     matching (5 disjoint ties) ~ edges + triangle. No dyad closes a
#     two-path, so the triangle column of the pseudo-likelihood design is all
#     zeros; R compares the observed triangle count (0) with its attainable
#     minimum (0) and fixes the coefficient at -Inf. The MPLE (exact: a
#     logistic regression) and the default MCMLE under eleven seeds.
# (b) A SINGLETON LEVEL: a 10-node network whose attribute `a` has levels
#     p (5 vertices), q (4) and r (1) ~ edges + nodemix("a"). The r.r cell has
#     no dyad at all (all-zero column) and two cells have no tie; R fixes all
#     three at -Inf. Dyad-independent: the MPLE is the exact MLE.
# (c) A DEGREE AT ITS MINIMUM WITH NO CHANGE: two disjoint 4-cliques (every
#     degree 3) ~ edges + degree(1). Adding or removing a tie never makes a
#     degree-1 vertex, so the column is all zeros; R fixes degree1 at -Inf.
#     The MPLE.
# (d) A STATISTIC THAT DOES NOT VARY (no bound reached): flomarriage with an
#     all-zero vertex attribute z ~ edges + nodecov("z"). R warns "Model
#     statistics 'nodecov.z' are not varying" and reports NA.
# (e) THE GOODREAU MODEL AS WRITTEN in the statnet tutorial: faux.mesa.high ~
#     edges + nodefactor("Grade") + nodematch("Grade", diff=TRUE) +
#     nodefactor("Race") + nodematch("Race", diff=TRUE) + nodefactor("Sex") +
#     nodematch("Sex") + gwesp(0.25, fixed=TRUE) + gwdegree(0.5, fixed=TRUE).
#     nodematch.Race.Black and .Other have no within-level tie; R fixes them
#     at -Inf. The MPLE (exact, pseudo-likelihood against pseudo-likelihood)
#     and the default MCMLE under five seeds; R's seed-to-seed sd is the
#     Monte-Carlo floor of the tolerance (as in mcmle_ergm.toml).

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm)
})

seed <- 20261007
set.seed(seed)
data(faux.mesa.high)
data(florentine)
fmh <- faux.mesa.high

quiet <- function(expr) {
  out <- NULL
  invisible(capture.output(out <- suppressMessages(suppressWarnings(expr)), type = "output"))
  out
}
se_of <- function(fit) sqrt(diag(vcov(fit)))
num <- function(x) paste(ifelse(is.na(x), "nan",
                                ifelse(is.finite(x), sprintf("%.17g", x),
                                       ifelse(x > 0, "inf", "-inf"))), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
tol4 <- function(sd, se, k) pmax(4 * sqrt(sd^2 + sd^2 / k), 0.1 * se)

# (a)
ma <- network.initialize(10, directed = FALSE)
add.edges(ma, seq(1, 9, 2), seq(2, 10, 2))
fa <- quiet(ergm(ma ~ edges + triangle, estimate = "MPLE"))
a_seeds <- c(seed, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
a_reps <- lapply(a_seeds, function(s) {
  set.seed(s); quiet(ergm(ma ~ edges + triangle, control = control.ergm(seed = s)))
})
a_edges <- sapply(a_reps, function(f) coef(f)[["edges"]])
a_se <- sapply(a_reps, function(f) se_of(f)[["edges"]])
a_tri <- sapply(a_reps, function(f) coef(f)[["triangle"]])
stopifnot(all(a_tri == -Inf))

# (b)
mb <- network.initialize(10, directed = FALSE)
b_el <- rbind(c(1, 2), c(1, 3), c(2, 4), c(3, 5), c(4, 5), c(1, 6), c(2, 7),
              c(6, 7), c(7, 8), c(8, 9), c(3, 9), c(5, 10))
add.edges(mb, b_el[, 1], b_el[, 2])
mb %v% "a" <- c("p", "p", "p", "p", "p", "q", "q", "q", "q", "r")
fb <- quiet(ergm(mb ~ edges + nodemix("a")))

# (c)
mc <- network.initialize(8, directed = FALSE)
c_el <- rbind(c(1, 2), c(1, 3), c(1, 4), c(2, 3), c(2, 4), c(3, 4),
              c(5, 6), c(5, 7), c(5, 8), c(6, 7), c(6, 8), c(7, 8))
add.edges(mc, c_el[, 1], c_el[, 2])
fc <- quiet(ergm(mc ~ edges + degree(1), estimate = "MPLE"))

# (d)
flo <- flomarriage
flo %v% "z" <- rep(0, network.size(flo))
fd <- quiet(ergm(flo ~ edges + nodecov("z")))

# (e)
f_e <- fmh ~ edges + nodefactor("Grade") + nodematch("Grade", diff = TRUE) +
  nodefactor("Race") + nodematch("Race", diff = TRUE) + nodefactor("Sex") +
  nodematch("Sex") + gwesp(0.25, fixed = TRUE) + gwdegree(0.5, fixed = TRUE)
fe <- quiet(ergm(f_e, estimate = "MPLE"))
# R's glm stops at its default epsilon = 1e-8, and on this 25-term design its
# standard errors are then ~2e-4 short of the converged ones (the
# coefficients agree to 1e-7). The converged values: the same logistic
# regression on the dyads the two -Inf statistics do not touch, at
# epsilon = 1e-14.
ed <- ergmMPLE(f_e, output = "matrix")
e_drop <- names(coef(fe))[is.infinite(coef(fe))]
e_keep <- rowSums(ed$predictor[, e_drop, drop = FALSE] != 0) == 0
e_X <- ed$predictor[e_keep, !(colnames(ed$predictor) %in% e_drop)]
e_glm <- glm(ed$response[e_keep] ~ e_X - 1, family = binomial, weights = ed$weights[e_keep],
             control = glm.control(epsilon = 1e-14, maxit = 100))
stopifnot(max(abs(coef(e_glm) - coef(fe)[is.finite(coef(fe))])) < 1e-6)
e_se_exact <- se_of(fe)
e_se_exact[is.finite(coef(fe))] <- sqrt(diag(vcov(e_glm)))
e_seeds <- c(seed, 101, 202, 303, 404)
e_reps <- lapply(e_seeds, function(s) {
  set.seed(s); quiet(ergm(f_e, control = control.ergm(seed = s)))
})
em <- t(sapply(e_reps, coef))
es <- t(sapply(e_reps, se_of))
fin <- is.finite(em[1, ])
e_mean <- colMeans(em); e_sd <- apply(em, 2, sd)
e_se <- colMeans(es)
e_sd[!fin] <- 0
e_tol <- tol4(e_sd, e_se, length(e_seeds))
e_tol[!fin] <- 0

cat('name = "boundary_ergm"\n\n')
cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/boundary_ergm.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('model_a = "10-node perfect matching ~ edges + triangle -- estimate=\\"MPLE\\", and the default MLE under eleven seeds"\n')
cat('model_b = "10-node network, attribute a = p x5, q x4, r x1 ~ edges + nodemix(\\"a\\") -- ergm() (logistic regression)"\n')
cat('model_c = "two disjoint 4-cliques ~ edges + degree(1) -- estimate=\\"MPLE\\""\n')
cat('model_d = "flomarriage, z = 0 on every vertex ~ edges + nodecov(\\"z\\") -- ergm()"\n')
cat('model_e = "faux.mesa.high ~ edges + nodefactor(\\"Grade\\") + nodematch(\\"Grade\\", diff=TRUE) + nodefactor(\\"Race\\") + nodematch(\\"Race\\", diff=TRUE) + nodefactor(\\"Sex\\") + nodematch(\\"Sex\\") + gwesp(0.25, fixed=TRUE) + gwdegree(0.5, fixed=TRUE) -- estimate=\\"MPLE\\", and the default MLE under five seeds"\n')
cat("\n[tolerance]\n")
cat("# (a)-(e) MPLE: the SAME logistic regression on both sides (after the same\n")
cat("# drop), solved to convergence (R IRLS ~1e-8, ERGM.jl Newton 1e-8). A failure\n")
cat("# is a bug; do not loosen. -Inf must match exactly, and NA (nan) is compared\n")
cat("# as 'not estimated' (NaN on the Julia side) position by position.\n")
for (k in c("a", "b", "c", "d")) cat(sprintf("%s_mple_coefficients = 1e-6\n%s_mple_std_errors = 1e-5\n", k, k))
cat("e_mple_coefficients = 1e-6\n")
cat("# ... except (e)'s standard errors AS SHIPPED: R's glm at its default epsilon\n")
cat("# stops ~2e-4 short of the converged values (e_mple_std_errors_exact, glm at\n")
cat("# epsilon = 1e-14, which ERGM.jl reproduces); the as-shipped numbers are\n")
cat("# compared at 5e-4 (a tolerance that measures R), the exact ones at 1e-6.\n")
cat("e_mple_std_errors_as_shipped = 5e-4\ne_mple_std_errors_exact = 1e-6\n")
cat("#\n")
cat("# (a), (e) MLE: one ERGM.jl MCMLE fit against R's seed mean, per coefficient:\n")
cat("# 4*sqrt(sd_R^2 + sd_R^2/k) floored at a tenth of R's standard error (the rule\n")
cat("# of mcmle_ergm.toml), emitted as a_mle_tolerance / e_mle_tolerance; the\n")
cat("# coefficients R fixes at -Inf must be -Inf (tolerance 0).\n\n")
cat("[values]\n")
cat(sprintf("a_terms = [%s]\n", strs(names(coef(fa)))))
cat(sprintf("a_mple_coefficients = [%s]\n", num(coef(fa))))
cat(sprintf("a_mple_std_errors = [%s]\n", num(se_of(fa))))
cat(sprintf("a_mle_edges_mean = %.17g\n", mean(a_edges)))
cat(sprintf("a_mle_edges_seed_sd = %.17g\n", sd(a_edges)))
cat(sprintf("a_mle_edges_se_mean = %.17g\n", mean(a_se)))
cat(sprintf("a_mle_tolerance = %.17g\n", tol4(sd(a_edges), mean(a_se), length(a_seeds))))
cat(sprintf("b_tails = [%s]\n", paste(b_el[, 1], collapse = ", ")))
cat(sprintf("b_heads = [%s]\n", paste(b_el[, 2], collapse = ", ")))
cat(sprintf("b_attribute = [%s]\n", strs(mb %v% "a")))
cat(sprintf("b_terms = [%s]\n", strs(names(coef(fb)))))
cat(sprintf("b_mple_coefficients = [%s]\n", num(coef(fb))))
cat(sprintf("b_mple_std_errors = [%s]\n", num(se_of(fb))))
cat(sprintf("c_tails = [%s]\n", paste(c_el[, 1], collapse = ", ")))
cat(sprintf("c_heads = [%s]\n", paste(c_el[, 2], collapse = ", ")))
cat(sprintf("c_terms = [%s]\n", strs(names(coef(fc)))))
cat(sprintf("c_mple_coefficients = [%s]\n", num(coef(fc))))
cat(sprintf("c_mple_std_errors = [%s]\n", num(se_of(fc))))
cat(sprintf("d_terms = [%s]\n", strs(names(coef(fd)))))
cat(sprintf("d_mple_coefficients = [%s]\n", num(coef(fd))))
cat(sprintf("d_mple_std_errors = [%s]\n", num(ifelse(is.na(coef(fd)), NA, se_of(fd)))))
cat(sprintf("e_terms = [%s]\n", strs(names(coef(fe)))))
cat(sprintf("e_mple_coefficients = [%s]\n", num(coef(fe))))
cat(sprintf("e_mple_std_errors_as_shipped = [%s]\n", num(se_of(fe))))
cat(sprintf("e_mple_std_errors_exact = [%s]\n", num(e_se_exact)))
cat(sprintf("e_mle_coef_mean = [%s]\n", num(e_mean)))
cat(sprintf("e_mle_coef_seed_sd = [%s]\n", num(e_sd)))
cat(sprintf("e_mle_se_mean = [%s]\n", num(e_se)))
cat(sprintf("e_mle_tolerance = [%s]\n", num(e_tol)))
