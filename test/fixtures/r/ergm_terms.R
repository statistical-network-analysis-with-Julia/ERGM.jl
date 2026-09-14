# Golden fixture: statnet `ergm` TERM PARITY — summary statistics, coefficient
# NAMES, an exact MPLE, and the errors R raises — on two bundled datasets.
#
# Regenerate from the package root (a few seconds; nothing here is Monte Carlo):
#
#   Rscript test/fixtures/r/ergm_terms.R > test/fixtures/ergm_terms.toml
#
# WHAT THIS FIXTURE PINS, AND WHY EACH THING IS PINNED AGAINST R RATHER THAN
# AGAINST A HAND LITERAL
#
# The 2026-09 panel (item 12) found three places where ERGM.jl's term layer
# quietly diverged from statnet for an R migrant:
#
#   1. `gwesp(0, fixed=TRUE)` -- the single most common applied specification --
#      was refused ("decay must be positive"), although the weight formula is
#      exact at decay 0 (it counts the edges with >= 1 shared partner).
#   2. `kstar`/`gwdegree` are undirected-only in R (the term "may not be used
#      with networks with directed==TRUE"); ERGM.jl silently computed OUT-stars
#      / out-degree weights under the undirected label on a directed network.
#      R's directed terms are `ostar`/`istar`/`gwodegree`/`gwidegree`.
#   3. A vertex without an attribute value was zero-filled; R refuses NA.
#
# and one the panel missed: R labels an integer-valued decay WITHOUT a decimal
# point (`gwesp.fixed.0`, `gwesp.fixed.1`, and `gwdegree` is labelled
# `gwdeg.fixed.<decay>`), so by-name coefficient comparison with a statnet fit
# broke on `gwesp.fixed.1.0` / `gwdegree.fixed.1.0`.
#
# The round-2 graders (2026-09) found three more, all pinned here since:
#
#   4. Directed `gwdsp(type="OSP"/"ISP")` was summed over UNORDERED dyads and
#      came out at exactly HALF of R's `dgwdsp` (R's `ddsp` sums every type
#      over ordered dyads). The samplike rows for dgwesp/dgwdsp ITP/OSP/ISP
#      make every directed shared-partner type an R number.
#   5. On a DIRECTED network R names the default gwesp/gwdsp
#      `gwesp.OTP.fixed.<decay>` / `gwdsp.OTP.fixed.<decay>` (the undirected
#      label has no type); ERGM.jl printed the undirected label. The samplike
#      names pin the directed labels, the flomarriage names the undirected.
#   6. `absdiff(pow=2)` is labelled `absdiff2.<attr>` (`absdiff.<attr>` only
#      for pow=1).
#
# The round-3 graders (2026-09) found three more, pinned here since:
#
#   7. `_boundary_columns` skipped a design column with entries of BOTH signs,
#      so an `edges + nodecov("x")` model whose observed nodecov sits at its
#      smallest attainable value (every dyad with a positive change is a
#      non-tie, every dyad with a negative change a tie) "converged" silently
#      to a coefficient of -42 with a standard error of 28,000. R's
#      `mple.existence` says "The MPLE does not exist!". The 8-node network
#      below is that case; the fixture records R's sentence.
#   8. After R's drop semantics ERGM.jl still counted the coefficient fixed
#      at -Inf in `dof`/AIC/BIC. R's `logLik.ergm` carries df = the number of
#      FINITE coefficients and nobs = the dyads the dropped term does not
#      touch (300 of 435 here), so BIC = -2 logLik + df * log(300); `nobs(fit)`
#      itself stays 435. The 30-node separated-nodematch network below (the
#      ERGM.jl testset's, formerly pinned to bare literals) records all of it.
#   9. The geodesic-distance GOF panel counted ORDERED pairs on an undirected
#      network (twice R) and dropped the unreachable ("Inf") row. R's
#      `gof(..., GOF=~distance)$obs.dist` on the directed samplike is recorded
#      here (the undirected flomarriage panel lives in flomarriage_ergm.R),
#      together with its idegree/odegree/esp observed panels.
#
# Every row below is therefore taken FROM R, not typed in:
#
#   (a) SUMMARY STATISTICS on flomarriage (undirected) for the decay-0 terms and
#       `degree(0:2)`, and on samplike (directed, bundled with ergm) for
#       `ostar`/`istar`/`gwidegree`/`gwodegree`/`triangle`. These are a
#       deterministic function of the observed graph: agreement is at machine
#       precision, and any disagreement is a bug in a term formula, full stop.
#   (b) The COEFFICIENT NAMES R emits for the same formulas, compared exactly.
#   (c) An MPLE of `edges + degree(0:2)` on flomarriage. The model is dyad-
#       DEPENDENT, so this is NOT an MLE comparison -- it is MPLE against MPLE:
#       both sides maximize the identical pseudo-likelihood (a logistic
#       regression on the change statistics), so the comparison is exact to
#       optimizer precision and pins the expansion of `Degree(0:2)` into three
#       design columns end to end. `vcov(fit)` of an `estimate="MPLE"` fit is
#       the inverse pseudo-information, which is what ERGM.jl's `mple` reports
#       as its (documented as anticonservative) Hessian standard errors.
#   (d) The ERROR STRINGS R raises for `kstar(2)` and `gwdegree(0.5, fixed=TRUE)`
#       on the directed samplike, and for `nodecov("x")` with one NA value:
#       recorded so the fixture documents that R refuses too, and the Julia
#       testset asserts that ERGM.jl errors wherever R errors.
#   (e) R's DROP semantics on a perfectly separated nodematch (30 nodes, 12
#       ties, every one between groups): coefficients (-Inf for the dropped
#       term), standard errors, logLik with its df and nobs attributes, AIC,
#       BIC and nobs(). The df/nobs rule is the point: df counts FINITE
#       coefficients and the BIC sample size is the dyads the dropped term
#       does not touch.
#   (f) A MIXED-SIGN design column at its boundary (8 nodes, x = -4..3, tie
#       iff x_i + x_j <= 0): the statistics and R's "The MPLE does not exist!"
#       warning, so the Julia side can assert it refuses where R refuses.
#   (f') A design separated only by a COMBINATION of columns (6 nodes, two
#       covariates x and z, tie iff (x_i + z_i) + (x_j + z_j) > 0): neither
#       nodecov column is at its attainable boundary, so the drop test cannot
#       see it, yet no finite MPLE exists. R's LP (`mple.existence`) warns;
#       ERGM.jl's asymptote test must fire on it.
#   (g) `gof(..., GOF=~distance + idegree + odegree + esp)` OBSERVED panels on
#       samplike, deterministic functions of the graph (R's distance panel
#       ends with the unreachable-pair count under the name "Inf").
#
# `samplike` (Sampson's monastery, 18 monks, 88 directed "like" ties) is
# exported as tail/head vectors so the directed rows are reproducible from the
# TOML alone; the Julia test rebuilds the network from them.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm)
})

seed <- 20260909   # nothing here is stochastic; recorded for provenance only
set.seed(seed)

data(florentine)
flo <- flomarriage
data(sampson)
sl <- samplike

quiet_ergm <- function(...) {
  fit <- NULL
  invisible(capture.output(fit <- ergm(...), type = "output"))
  fit
}

num <- function(x) paste(ifelse(is.finite(x), sprintf("%.17g", x),
                                ifelse(x > 0, "inf", "-inf")), collapse = ", ")
ints <- function(x) paste(sprintf("%d", as.integer(x)), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
# A TOML basic string: escape backslashes and double quotes; R's fancy quotes
# around term names are kept verbatim (UTF-8).
tstr <- function(x) sprintf('"%s"', gsub('"', '\\\\"', gsub("\\\\", "\\\\\\\\", x)))
err_of <- function(expr) tryCatch({ expr; "" }, error = function(e) conditionMessage(e))

# --- (a)/(b) flomarriage: decay-0 GW terms and degree(0:2) --------------------
f_flo <- flo ~ edges + gwesp(0, fixed = TRUE) + gwdsp(0, fixed = TRUE) +
  gwdegree(0, fixed = TRUE) + gwdegree(1, fixed = TRUE) + degree(0:2) +
  absdiff("wealth", pow = 2) + absdiff("wealth")
sum_flo <- summary(f_flo)

# --- (c) MPLE of a dyad-dependent model: pseudo-likelihood vs pseudo-likelihood
f_mple <- flo ~ edges + degree(0:2)
fit_mple <- suppressMessages(quiet_ergm(f_mple, estimate = "MPLE"))
se_mple <- sqrt(diag(vcov(fit_mple)))

# --- (a)/(b) samplike: the directed star / degree-weighted terms --------------
# The directed shared-partner terms come AFTER ttriple so the first eight
# positions stay what the 0.2 testset indexes: gwesp/gwdsp with their default
# type (R's directed label carries `OTP`), then the typed dgwesp/dgwdsp rows.
f_sl <- sl ~ edges + mutual + ostar(2) + istar(2) + gwidegree(0.5, fixed = TRUE) +
  gwodegree(0.5, fixed = TRUE) + triangle + ttriple +
  gwesp(0.5, fixed = TRUE) + gwdsp(0.5, fixed = TRUE) +
  dgwesp(0.5, fixed = TRUE, type = "ITP") + dgwesp(0.5, fixed = TRUE, type = "OSP") +
  dgwesp(0.5, fixed = TRUE, type = "ISP") +
  dgwdsp(0.5, fixed = TRUE, type = "ITP") + dgwdsp(0.5, fixed = TRUE, type = "OSP") +
  dgwdsp(0.5, fixed = TRUE, type = "ISP")
sum_sl <- summary(f_sl)
el <- as.edgelist(sl)

# --- (d) what R refuses -------------------------------------------------------
err_kstar <- err_of(summary(sl ~ kstar(2)))
err_gwdeg <- err_of(summary(sl ~ gwdegree(0.5, fixed = TRUE)))
x <- flo %v% "wealth"
x[3] <- NA
flo %v% "x" <- x
err_na <- err_of(summary(flo ~ nodecov("x")))

# --- (e) R's drop semantics on a perfectly separated nodematch ---------------
# The ERGM.jl testset's 30-node network: 12 ties, group = ("A","B","C")[v mod 3],
# every tie between different groups, so nodematch.group = 0 is at its
# smallest attainable value.
sep_el <- rbind(c(1, 2), c(1, 3), c(2, 3), c(2, 4), c(3, 4), c(3, 5), c(4, 5),
                c(5, 6), c(6, 7), c(7, 8), c(8, 9), c(9, 10))
sep <- network.initialize(30, directed = FALSE)
add.edges(sep, sep_el[, 1], sep_el[, 2])
sep %v% "group" <- c("A", "B", "C")[((1:30 - 1) %% 3) + 1]
fit_sep <- suppressMessages(suppressWarnings(quiet_ergm(sep ~ edges + nodematch("group"))))
ll_sep <- logLik(fit_sep)

# --- (f) a mixed-sign boundary column: R's "The MPLE does not exist!" --------
# 8 nodes, x = -4..3, tie iff x_i + x_j <= 0. The nodecov change statistic
# x_i + x_j takes both signs, and every dyad with a positive change is a
# non-tie while every dyad with a negative (or zero) change is a tie, so the
# observed nodecov (-50) is the smallest value the statistic can attain.
mix <- network.initialize(8, directed = FALSE)
mix_x <- -4:3
for (i in 1:7) for (j in (i + 1):8) if (mix_x[i] + mix_x[j] <= 0) add.edges(mix, i, j)
mix %v% "x" <- mix_x
sum_mix <- summary(mix ~ edges + nodecov("x"))
warnings_of <- function(expr) {
  w <- character(0)
  withCallingHandlers(
    tryCatch(expr, error = function(e) NULL),
    warning = function(x) { w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") })
  w
}
warn_mix <- warnings_of(quiet_ergm(mix ~ edges + nodecov("x"), estimate = "MPLE"))
warn_mix_mple <- warn_mix[grepl("MPLE does not exist", warn_mix)][1]
if (is.na(warn_mix_mple)) stop("R did not warn that the MPLE does not exist on the mixed-sign network")

# --- (f') separated by a combination of two nodecov columns --------------------
sep2 <- network.initialize(6, directed = FALSE)
sep2_x <- c(1, 2, 1, 2, 3, -2)
sep2_z <- c(-1, -2, 3, 0, -3, -1)
for (i in 1:5) for (j in (i + 1):6)
  if ((sep2_x[i] + sep2_z[i]) + (sep2_x[j] + sep2_z[j]) > 0) add.edges(sep2, i, j)
sep2 %v% "x" <- sep2_x
sep2 %v% "z" <- sep2_z
sum_sep2 <- summary(sep2 ~ edges + nodecov("x") + nodecov("z"))
warn_sep2 <- warnings_of(quiet_ergm(sep2 ~ edges + nodecov("x") + nodecov("z"), estimate = "MPLE"))
warn_sep2_mple <- warn_sep2[grepl("MPLE does not exist", warn_sep2)][1]
if (is.na(warn_sep2_mple)) stop("R did not warn that the MPLE does not exist on the combination-separated network")

# --- (g) gof observed panels on samplike (deterministic) -----------------------
fit_sl_eo <- suppressMessages(quiet_ergm(sl ~ edges, estimate = "MPLE"))
gof_sl <- suppressMessages(gof(fit_sl_eo, GOF = ~distance + idegree + odegree + esp,
                               control = control.gof.ergm(nsim = 2)))

cat('name = "ergm_terms"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/ergm_terms.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('datasets = "ergm::flomarriage (Padgett: 16 Florentine families, 20 undirected marriage ties, wealth covariate); ergm::samplike (Sampson monastery, 18 monks, 88 directed like-ties, from data(sampson))"\n')
cat('model_flo_summary = "flomarriage ~ edges + gwesp(0, fixed=TRUE) + gwdsp(0, fixed=TRUE) + gwdegree(0, fixed=TRUE) + gwdegree(1, fixed=TRUE) + degree(0:2) + absdiff(\\"wealth\\", pow=2) + absdiff(\\"wealth\\") -- summary() only"\n')
cat('model_flo_mple = "flomarriage ~ edges + degree(0:2) -- ergm(estimate=\\"MPLE\\"); dyad-dependent, so pseudo-likelihood on BOTH sides"\n')
cat('model_separated = "30-node undirected network, 12 ties between different groups (group = (A,B,C)[v mod 3]) ~ edges + nodematch(\\"group\\") -- ergm() with its default drop=TRUE; nodematch.group = 0 is at its smallest attainable value"\n')
cat('model_mixed_sign = "8-node undirected network, x = -4..3, tie iff x_i + x_j <= 0 ~ edges + nodecov(\\"x\\") -- ergm(estimate=\\"MPLE\\"); nodecov.x = -50 is at its smallest attainable value although its change statistic takes both signs"\n')
cat('model_combination_separated = "6-node undirected network, x = (1,2,1,2,3,-2), z = (-1,-2,3,0,-3,-1), tie iff (x_i + z_i) + (x_j + z_j) > 0 ~ edges + nodecov(\\"x\\") + nodecov(\\"z\\") -- ergm(estimate=\\"MPLE\\"); no single statistic is at its attainable boundary, but the design is separated by nodecov.x + nodecov.z"\n')
cat('model_samplike_gof = "samplike ~ edges (MPLE) -- gof(GOF = ~distance + idegree + odegree + esp) observed panels only"\n')
cat('model_samplike_summary = "samplike ~ edges + mutual + ostar(2) + istar(2) + gwidegree(0.5, fixed=TRUE) + gwodegree(0.5, fixed=TRUE) + triangle + ttriple + gwesp(0.5, fixed=TRUE) + gwdsp(0.5, fixed=TRUE) + dgwesp(0.5, fixed=TRUE, type=ITP/OSP/ISP) + dgwdsp(0.5, fixed=TRUE, type=ITP/OSP/ISP) -- summary() only; the directed gwesp/gwdsp labels carry the type"\n')
cat("\n")

cat("[tolerance]\n")
cat("# Summary statistics are a DETERMINISTIC function of the observed graph --\n")
cat("# no estimator, no simulation. Machine precision; any disagreement is a bug\n")
cat("# in a term formula (or in the Degree(0:2) expansion), full stop.\n")
cat("flo_summary = 1e-9\n")
cat("samplike_summary = 1e-9\n")
cat("#\n")
cat("# MPLE vs MPLE. edges + degree(0:2) is dyad-DEPENDENT, so this is not an\n")
cat("# MLE comparison and says nothing about the MLE -- but the pseudo-likelihood\n")
cat("# is the SAME strictly-concave logistic-regression objective on both sides\n")
cat("# (R: glm on the change statistics via estimate=\"MPLE\"; ERGM.jl: Newton on\n")
cat("# the compressed change-statistic design), and both solve it to convergence.\n")
cat("# The only difference two correct implementations can have is optimizer\n")
cat("# termination (R's IRLS ~1e-8, ERGM.jl's Newton 1e-8). 1e-6 is well inside\n")
cat("# both and ~1e-6 of the smallest standard error. A failure is a BUG in the\n")
cat("# design (the Degree(0:2) expansion, the change statistics) -- do not loosen.\n")
cat("flo_mple_coefficients = 1e-6\n")
cat("flo_mple_std_errors = 1e-6\n")
cat("#\n")
cat("# Drop semantics (separated nodematch): the finite coefficient is the exact\n")
cat("# logistic regression on the 300 dyads the dropped term does not touch and\n")
cat("# the dropped one is -Inf on both sides (isapprox(-Inf, -Inf) holds), so\n")
cat("# the optimizer-precision argument above applies unchanged. logLik, AIC and\n")
cat("# BIC follow from them in closed form; 1e-5 covers the rounding of a sum of\n")
cat("# 300 log-probabilities. df and nobs are integers, compared exactly.\n")
cat("sep_coefficients = 1e-6\n")
cat("sep_std_errors = 1e-6\n")
cat("sep_loglik = 1e-5\n")
cat("sep_aic = 1e-5\n")
cat("sep_bic = 1e-5\n")
cat("#\n")
cat("# The mixed-sign statistics and the gof observed panels are deterministic\n")
cat("# functions of the observed graph: machine precision.\n")
cat("mix_summary = 1e-9\n")
cat("sep2_summary = 1e-9\n")
cat("samplike_gof_distance = 1e-9\n")
cat("samplike_gof_idegree = 1e-9\n")
cat("samplike_gof_odegree = 1e-9\n")
cat("samplike_gof_esp = 1e-9\n\n")

cat("[values]\n")
cat("# --- flomarriage (undirected): decay-0 GW terms, R's labels, degree(0:2),\n")
cat("# absdiff(pow=2) (labelled absdiff2.wealth) and absdiff (absdiff.wealth) ---\n")
cat(sprintf("flo_summary_names = [%s]\n", strs(names(sum_flo))))
cat(sprintf("flo_summary = [%s]\n", num(sum_flo)))
cat("\n# --- flomarriage MPLE of edges + degree(0:2): coefficients and inverse-\n")
cat("# pseudo-information standard errors (vcov of an estimate=\"MPLE\" fit) ------\n")
cat(sprintf("flo_mple_terms = [%s]\n", strs(names(coef(fit_mple)))))
cat(sprintf("flo_mple_coefficients = [%s]\n", num(coef(fit_mple))))
cat(sprintf("flo_mple_std_errors = [%s]\n", num(se_mple)))
cat("\n# --- samplike (directed): edge list (tail -> head, 1-based) and statistics.\n")
cat("# `ttriple` has no separate ERGM.jl term: `Triangle()` on a directed network\n")
cat("# is statnet's `triangle` (= ttriple + ctriple), compared here; `ttriple` is\n")
cat("# recorded for reference only. Positions 9-16 are the directed shared-partner\n")
cat("# terms: gwesp/gwdsp with the default type (R's directed labels\n")
cat("# gwesp.OTP.fixed.0.5 / gwdsp.OTP.fixed.0.5) and the typed dgwesp/dgwdsp rows\n")
cat("# (ITP, OSP, ISP) -- every one summed over ORDERED dyads in R's C code. ------\n")
cat(sprintf("samplike_n = %d\n", network.size(sl)))
cat(sprintf("samplike_tails = [%s]\n", ints(el[, 1])))
cat(sprintf("samplike_heads = [%s]\n", ints(el[, 2])))
cat(sprintf("samplike_summary_names = [%s]\n", strs(names(sum_sl))))
cat(sprintf("samplike_summary = [%s]\n", num(sum_sl)))
cat("\n# --- what R refuses (conditionMessage of the error), so the fixture documents\n")
cat("# that ERGM.jl's refusals mirror statnet's -------------------------------\n")
cat(sprintf("r_error_kstar_directed = %s\n", tstr(err_kstar)))
cat(sprintf("r_error_gwdegree_directed = %s\n", tstr(err_gwdeg)))
cat(sprintf("r_error_nodecov_na = %s\n", tstr(err_na)))
cat("\n# --- (e) R's drop=TRUE on the perfectly separated nodematch (30 nodes):\n")
cat("# the -Inf coefficient, SE 0, and the df/nobs rule behind AIC/BIC -- df is\n")
cat("# the number of FINITE coefficients, logLik's nobs the dyads the dropped\n")
cat("# term does not touch (300 of 435), while nobs(fit) stays every dyad ------\n")
cat(sprintf("sep_terms = [%s]\n", strs(names(coef(fit_sep)))))
cat(sprintf("sep_coefficients = [%s]\n", num(coef(fit_sep))))
cat(sprintf("sep_std_errors = [%s]\n", num(sqrt(diag(vcov(fit_sep))))))
cat(sprintf("sep_loglik = %.17g\n", as.numeric(ll_sep)))
cat(sprintf("sep_df = %d\n", as.integer(attr(ll_sep, "df"))))
cat(sprintf("sep_loglik_nobs = %d\n", as.integer(attr(ll_sep, "nobs"))))
cat(sprintf("sep_nobs = %d\n", as.integer(nobs(fit_sep))))
cat(sprintf("sep_aic = %.17g\n", AIC(fit_sep)))
cat(sprintf("sep_bic = %.17g\n", BIC(fit_sep)))
cat("\n# --- (f) the mixed-sign boundary column (8 nodes, x = -4..3, tie iff\n")
cat("# x_i + x_j <= 0): statistics, and the warning R's mple.existence raises ----\n")
cat(sprintf("mix_n = %d\n", network.size(mix)))
cat(sprintf("mix_x = [%s]\n", ints(mix_x)))
cat(sprintf("mix_summary_names = [%s]\n", strs(names(sum_mix))))
cat(sprintf("mix_summary = [%s]\n", num(sum_mix)))
cat(sprintf("r_warning_mple_nonexistent = %s\n", tstr(warn_mix_mple)))
cat("\n# --- (f') separated by a COMBINATION of columns (6 nodes; no single\n")
cat("# column at its boundary): statistics and R's mple.existence warning -------\n")
cat(sprintf("sep2_n = %d\n", network.size(sep2)))
cat(sprintf("sep2_x = [%s]\n", ints(sep2_x)))
cat(sprintf("sep2_z = [%s]\n", ints(sep2_z)))
cat(sprintf("sep2_summary_names = [%s]\n", strs(names(sum_sep2))))
cat(sprintf("sep2_summary = [%s]\n", num(sum_sep2)))
cat(sprintf("r_warning_mple_nonexistent_combination = %s\n", tstr(warn_sep2_mple)))
cat("\n# --- (g) gof observed panels on samplike: distance over ORDERED pairs\n")
cat("# (directed), the last entry the unreachable pairs R labels \"Inf\";\n")
cat("# idegree/odegree over 0:(n-1); esp (OTP) over 0:(n-2) -------------------\n")
cat(sprintf("samplike_gof_distance_labels = [%s]\n", strs(names(gof_sl$obs.dist))))
cat(sprintf("samplike_gof_distance = [%s]\n", num(gof_sl$obs.dist)))
cat(sprintf("samplike_gof_idegree = [%s]\n", num(gof_sl$obs.ideg)))
cat(sprintf("samplike_gof_odegree = [%s]\n", num(gof_sl$obs.odeg)))
cat(sprintf("samplike_gof_esp_labels = [%s]\n", strs(names(gof_sl$obs.esp))))
cat(sprintf("samplike_gof_esp = [%s]\n", num(gof_sl$obs.esp)))
