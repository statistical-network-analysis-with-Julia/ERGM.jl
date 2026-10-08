# Golden fixture: statnet `ergm` with offset() terms -- coefficients fixed,
# not estimated.
#
# Regenerate from the package root (~1 min):
#
#   Rscript test/fixtures/r/offset_ergm.R > test/fixtures/offset_ergm.toml
#
# (a) A STRUCTURAL ZERO: faux.mesa.high ~ edges + nodematch("Grade") +
#     offset(nodemix("Grade", levels2=c(2,4))) with offset.coef = c(-Inf,-Inf)
#     -- grade-7 students have no tie to grade 8 or 9, and the model says they
#     cannot. Dyad-independent, so ergm() fits it by logistic regression on
#     the dyads the offset leaves free: the MPLE is the exact MLE and the
#     comparison is at optimizer precision.
# (b) A FINITE OFFSET: faux.mesa.high ~ offset(edges) + nodematch("Grade")
#     with offset.coef = -5. Dyad-independent: exact as (a).
# (d) A FORCED TIE, offset.coef = +Inf: a 10-node network whose group A
#     (vertices 1-4) is a complete clique ~ edges + nodecov("x") +
#     offset(nodematch("g", diff=TRUE, levels=1)) at +Inf -- every A-A tie is
#     forced. Dyad-independent: exact.
# (c) A DYAD-DEPENDENT MODEL WITH AN OFFSET: flomarriage ~ offset(edges) +
#     gwesp(0.5, fixed=TRUE), offset.coef = -1.7, fitted by MCMLE at ergm()'s
#     defaults under eleven seeds; R's seed-to-seed sd is the Monte-Carlo
#     floor of the tolerance (as in mcmle_ergm.toml).

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
se_of <- function(fit) sqrt(diag(vcov(fit)))
num <- function(x) paste(ifelse(is.finite(x), sprintf("%.17g", x),
                                ifelse(x > 0, "inf", "-inf")), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")

fa <- quiet(ergm(fmh ~ edges + nodematch("Grade") + offset(nodemix("Grade", levels2 = c(2, 4))),
                 offset.coef = c(-Inf, -Inf)))
fb <- quiet(ergm(fmh ~ offset(edges) + nodematch("Grade"), offset.coef = -5))
la <- logLik(fa); lb <- logLik(fb)
# (b) has a closed form: one estimated coefficient on a 0/1 statistic, so the
# score equation is 163 = N_m p with p = logistic(-5 + beta) over the N_m
# same-grade dyads, and the information is 163 (1 - p)
grade <- fmh %v% "Grade"
n_same <- sum(outer(grade, grade, "==")[upper.tri(diag(length(grade)))])
nm_obs <- summary(fmh ~ nodematch("Grade"))[[1]]
p_m <- nm_obs / n_same
b_exact_coef <- log(p_m / (1 - p_m)) + 5
b_exact_se <- 1 / sqrt(nm_obs * (1 - p_m))

dnet <- network.initialize(10, directed = FALSE)
d_el <- rbind(c(1, 2), c(1, 3), c(1, 4), c(2, 3), c(2, 4), c(3, 4), c(4, 5), c(5, 6),
              c(6, 7), c(2, 8), c(8, 9), c(9, 10), c(5, 9))
add.edges(dnet, d_el[, 1], d_el[, 2])
dnet %v% "g" <- ifelse(1:10 <= 4, "A", "B")
dnet %v% "x" <- as.numeric(1:10)
fd <- quiet(ergm(dnet ~ edges + nodecov("x") + offset(nodematch("g", diff = TRUE, levels = 1)),
                 offset.coef = Inf))
ld <- logLik(fd)

f_c <- flomarriage ~ offset(edges) + gwesp(0.5, fixed = TRUE)
rep_seeds <- c(seed, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
reps <- lapply(rep_seeds, function(s) {
  set.seed(s); quiet(ergm(f_c, offset.coef = -1.7, control = control.ergm(seed = s)))
})
cm <- t(sapply(reps, coef))
sm <- t(sapply(reps, se_of))
c_mean <- colMeans(cm); c_sd <- apply(cm, 2, sd)
s_mean <- colMeans(sm); s_sd <- apply(sm, 2, sd)
tol4 <- function(sd, se) pmax(4 * sqrt(sd^2 + sd^2 / length(rep_seeds)), 0.1 * se)

cat('name = "offset_ergm"\n\n')
cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/offset_ergm.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('model_a = "faux.mesa.high ~ edges + nodematch(\\"Grade\\") + offset(nodemix(\\"Grade\\", levels2=c(2,4))), offset.coef=c(-Inf,-Inf) -- ergm() (logistic regression: MPLE = MLE)"\n')
cat('model_b = "faux.mesa.high ~ offset(edges) + nodematch(\\"Grade\\"), offset.coef=-5"\n')
cat('model_c = "flomarriage ~ offset(edges) + gwesp(0.5, fixed=TRUE), offset.coef=-1.7 -- MCMLE at ergm() defaults, eleven seeds"\n')
cat("\n[tolerance]\n")
cat("# (a), (b): dyad-independent -- the SAME logistic regression on both sides,\n")
cat("# solved to convergence (R IRLS ~1e-8, ERGM.jl Newton 1e-8). A failure is a\n")
cat("# bug; do not loosen. The offset coefficients are fixed (-Inf, -5) and the\n")
cat("# SEs of the offsets are 0 on both sides.\n")
cat("a_coefficients = 1e-6\na_std_errors = 1e-6\na_loglik = 1e-5\na_aic = 1e-5\na_bic = 1e-5\n")
cat("b_coefficients = 1e-6\nb_loglik = 1e-5\nb_aic = 1e-5\nb_bic = 1e-5\n")
cat("# ... except (b)'s standard error AS SHIPPED: R's glm stops ~2e-6 short of the\n")
cat("# closed-form value (b_exact_*), which ERGM.jl reproduces to ~1e-16; the\n")
cat("# as-shipped number is compared at 1e-5 (a tolerance that measures R), the\n")
cat("# exact one at 1e-9 (the tolerance that measures Julia).\n")
cat("b_std_errors = 1e-5\nb_exact_coefficient = 1e-9\nb_exact_std_error = 1e-9\n")
cat("# (d): the same argument; R's glm slack shows in the SEs (2e-7), hence 1e-5.\n")
cat("d_coefficients = 1e-6\nd_std_errors = 1e-5\nd_loglik = 1e-5\nd_aic = 1e-5\nd_bic = 1e-5\n")
cat("#\n")
cat("# (c): one ERGM.jl MCMLE fit against R's eleven-seed mean, per coefficient:\n")
cat("# 4*sqrt(sd_R^2 + sd_R^2/11) floored at a tenth of R's standard error (the\n")
cat("# rule of mcmle_ergm.toml), emitted as c_coef_tolerance / c_se_tolerance.\n\n")
cat("[values]\n")
cat(sprintf("a_terms = [%s]\n", strs(names(coef(fa)))))
cat(sprintf("a_coefficients = [%s]\n", num(coef(fa))))
cat(sprintf("a_std_errors = [%s]\n", num(se_of(fa))))
cat(sprintf("a_loglik = %.17g\n", as.numeric(la)))
cat(sprintf("a_df = %d\n", as.integer(attr(la, "df"))))
cat(sprintf("a_aic = %.17g\n", AIC(fa)))
cat(sprintf("a_bic = %.17g\n", BIC(fa)))
cat(sprintf("a_nobs = %d\n", as.integer(nobs(fa))))
cat(sprintf("b_terms = [%s]\n", strs(names(coef(fb)))))
cat(sprintf("b_coefficients = [%s]\n", num(coef(fb))))
cat(sprintf("b_std_errors = [%s]\n", num(se_of(fb))))
cat(sprintf("b_loglik = %.17g\n", as.numeric(lb)))
cat(sprintf("b_df = %d\n", as.integer(attr(lb, "df"))))
cat(sprintf("b_exact_coefficient = %.17g\n", b_exact_coef))
cat(sprintf("b_exact_std_error = %.17g\n", b_exact_se))
cat(sprintf("b_aic = %.17g\n", AIC(fb)))
cat(sprintf("b_bic = %.17g\n", BIC(fb)))
cat(sprintf("d_tails = [%s]\n", paste(d_el[, 1], collapse = ", ")))
cat(sprintf("d_heads = [%s]\n", paste(d_el[, 2], collapse = ", ")))
cat(sprintf("d_terms = [%s]\n", strs(names(coef(fd)))))
cat(sprintf("d_coefficients = [%s]\n", num(coef(fd))))
cat(sprintf("d_std_errors = [%s]\n", num(se_of(fd))))
cat(sprintf("d_loglik = %.17g\n", as.numeric(ld)))
cat(sprintf("d_df = %d\n", as.integer(attr(ld, "df"))))
cat(sprintf("d_aic = %.17g\n", AIC(fd)))
cat(sprintf("d_bic = %.17g\n", BIC(fd)))
cat(sprintf("c_terms = [%s]\n", strs(names(coef(reps[[1]])))))
cat(sprintf("c_coef_mean = [%s]\n", num(c_mean)))
cat(sprintf("c_coef_seed_sd = [%s]\n", num(c_sd)))
cat(sprintf("c_se_mean = [%s]\n", num(s_mean)))
cat(sprintf("c_se_seed_sd = [%s]\n", num(s_sd)))
cat(sprintf("c_coef_tolerance = [%s]\n", num(tol4(c_sd, s_mean))))
cat(sprintf("c_se_tolerance = [%s]\n", num(tol4(s_sd, s_mean))))
