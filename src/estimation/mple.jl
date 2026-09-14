"""
Maximum Pseudo-Likelihood Estimation (MPLE) for ERGMs.

MPLE treats each potential edge as an independent observation and fits
a logistic regression of the dyad indicators on the add-direction change
statistics, each computed conditional on the rest of the observed network.
This is fast but may be biased for models with strong dependencies.

The logistic fit itself is the ecosystem's shared kernel: the binomial-row
form of `Networks.logistic_derivatives` maximized by `Networks.newton_fit`
(panel 2026-09, item 14). ERGM.jl adds only the design construction.
"""

# Numerically stable log(1 + exp(x)) — used by `_dyad_independent_logZ`
_log1pexp(x::Float64) = x > 35 ? x : (x < -35 ? exp(x) : log1p(exp(x)))

"""
    _z_pvalues

Deprecated `public` `const` alias of `Networks.z_pvalues` — the ONE z → p
helper of the ecosystem (erfc-based, floored at floatmin, NaN-aware);
removed once TERGM/ERGMMulti import `Networks.z_pvalues` themselves (panel
2026-09, item 13).

# Example
```julia
using ERGM
ERGM._z_pvalues === z_pvalues           # true
z_pvalues([1.96])[1] < 0.051            # true
```
"""
const _z_pvalues = z_pvalues

"""
    _mple_data(net, terms::TermSet, directed::Bool)

Build compressed MPLE data: unique change-statistic rows with the number of
dyads (`n_tot`) and the number of observed edges (`n_one`) sharing each row.
Compressing identical rows keeps memory O(unique rows) instead of O(n²).

Dyads masked as missing (see `Networks.set_missing_dyad!`) are excluded: their
tie status is unobserved, so they contribute no response row. Their face
values still enter the *predictors* of the remaining dyads, since change
statistics are computed conditional on the rest of the observed network.

The row table is keyed on an `NTuple{p,Float64}` — the change-statistic
tuple built term by term by the generated `_change_stat_tuple` lives on the
stack for ANY `p` (a `Base.map` over the term tuple boxed every value from 32
terms on) — so the sweep allocates O(unique rows) (the Dict and the returned
arrays), not a fresh `Vector{Float64}` per dyad (panel 2026-09, item 26).
Pinned by the "MPLE design build allocates O(unique rows)" testset, at 3 and
at 36 statistics.
"""
function _mple_data(net, terms::TermSet, directed::Bool)
    p = length(terms)
    K = NTuple{p, Float64}
    rows = _mple_rows!(Dict{K, Tuple{Float64, Float64}}(), net, terms, directed)

    n_rows = length(rows)
    X = Matrix{Float64}(undef, n_rows, p)
    n_tot = Vector{Float64}(undef, n_rows)
    n_one = Vector{Float64}(undef, n_rows)

    for (r, (x, (tot, one))) in enumerate(rows)
        for c in 1:p
            X[r, c] = x[c]
        end
        n_tot[r] = tot
        n_one[r] = one
    end

    return X, n_tot, n_one
end

# Function barrier: specialized on the key type `K`, so the Dict operations
# and the tuple of change statistics are statically typed.
function _mple_rows!(rows::Dict{K, Tuple{Float64, Float64}}, net, terms::TermSet,
                     directed::Bool) where {K}
    n = nv(net)
    for i in 1:n
        j_range = directed ? (1:n) : ((i+1):n)
        for j in j_range
            i == j && continue
            is_missing_dyad(net, i, j) && continue

            x = _change_stat_tuple(terms, net, i, j)::K
            n_tot, n_one = get(rows, x, (0.0, 0.0))
            rows[x] = (n_tot + 1.0, n_one + (has_edge(net, i, j) ? 1.0 : 0.0))
        end
    end
    return rows
end

"""
    _boundary_columns(X, n_tot, n_one) -> Vector{Tuple{Int,Symbol}}

The design columns whose observed statistic sits at the boundary of its
attainable range, read off the compressed pseudo-likelihood design — R
ergm's attainable-range test (`ergm.checkextreme.model`). Under the
pseudo-likelihood each dyad `r` contributes `X[r, j]` to statistic `j` if
tied, so the smallest value the statistic can attain is `Σ_r n_tot[r] ·
min(0, X[r, j])` and the observed value sits AT it iff every dyad with
`X[r, j] > 0` is a non-tie (`n_one[r] == 0`) and every dyad with
`X[r, j] < 0` is a tie (`n_one[r] == n_tot[r]`); the mirror condition is the
largest attainable value. `public`: ERGMRank, ERGMMulti and TERGM apply the
same test to their own designs. The sign pattern of the column is irrelevant: a
`nodecov` whose change statistic `x_i + x_j` takes both signs is at its
minimum whenever the positive-change dyads are all empty and the
negative-change dyads all tied (round-3 finding: such a column used to be
skipped as "mixed-sign, no monotone direction", and Newton then "converged"
on the flat asymptote at −42 with a standard error of 28,000 — R says "The
MPLE does not exist!").

For such a column the pseudo-log-likelihood increases monotonically as the
coefficient goes to `-Inf` (`:min` — the statistic is at its smallest
attainable value) or `+Inf` (`:max`) — its gradient
`Σ_r X[r, j] (n_one[r] − n_tot[r] p_r)` keeps one sign at every θ — so no
finite MPLE exists: R ergm's "Observed statistic(s) … are at their smallest
attainable values. Their coefficients will be fixed at -Inf." (the
`drop=TRUE` default of `control.ergm`). A perfectly separated `nodematch` —
zero within-group ties — is the textbook case.

# Example
```julia
using ERGM
# Two compressed rows: 3 dyads with change statistic +1, 2 dyads with −1
X = reshape([1.0, -1.0], 2, 1)
ERGM._boundary_columns(X, [3.0, 2.0], [0.0, 2.0])   # [(1, :min)]: +1 dyads empty, −1 dyads tied
ERGM._boundary_columns(X, [3.0, 2.0], [3.0, 0.0])   # [(1, :max)]
ERGM._boundary_columns(X, [3.0, 2.0], [1.0, 1.0])   # empty: strictly inside its range
```
"""
function _boundary_columns(X::AbstractMatrix, n_tot::AbstractVector, n_one::AbstractVector)
    out = Tuple{Int,Symbol}[]
    for j in 1:size(X, 2)
        at_min = at_max = true      # observed statistic at its smallest / largest value
        any_nz = false
        for r in 1:size(X, 1)
            x = X[r, j]
            x == 0 && continue
            any_nz = true
            empty = n_one[r] == 0
            full = n_one[r] == n_tot[r]
            if x > 0
                empty || (at_min = false)
                full || (at_max = false)
            else
                full || (at_min = false)
                empty || (at_max = false)
            end
            at_min || at_max || break
        end
        any_nz || continue
        if at_min
            push!(out, (j, :min))
        elseif at_max
            push!(out, (j, :max))
        end
    end
    return out
end

"""
    _separated(f, X, n_tot, n_one, θ, se) -> Bool

Whether a Newton "solution" `θ` of the pseudo-likelihood `f` (the
`logistic_derivatives` closure) is really a point on a flat asymptote — the
MPLE does not exist by complete or quasi-complete separation of a kind
`_boundary_columns` cannot see (a linear combination of columns that
perfectly predicts the ties). R's `mple.existence` decides this with an
exact linear program (a direction `β` with `y·Xβ ≥ 0` on every dyad and an
unbounded objective); ERGM.jl has no LP solver, so it tests the two
signatures an asymptote leaves on the Newton iteration, BOTH of which must
hold:

1. a compressed row whose response is perfectly predicted (all non-ties or
   all ties) at a fitted probability within 1e-8 of 0 or 1 — a linear
   predictor beyond ±18.4, which a finite maximizer never needs; and
2. the next Newton step `−H⁻¹g` is still large relative to `θ` (> 1e-3 of
   `‖θ‖`), or the Hessian is not invertible (NaN standard errors): on the
   asymptote Newton keeps moving by O(1/scale) per step while the objective
   gains less than `tol`, whereas at a finite maximum quadratic convergence
   has shrunk the step to ~1e-8.

Either signature alone has benign explanations (an extreme but well-fitted
dyad on a large network; a coefficient that happens to be ≈ 0); together
they are the asymptote. The grader's mixed-sign `nodecov` design (Newton
"converged" at −42, standard error 28,000) satisfies both. `public`: the
variants' MPLEs (ERGMMulti, ERGMRank) apply the same verdict; an offset
contribution is passed as an extra design column with a fixed coefficient.

# Example
```julia
using ERGM
# A well-posed design: an extreme (η = 30·θ₂) but well-determined dyad class
X = [1.0 0.0; 1.0 30.0]
f = ERGM.logistic_derivatives(X, [100.0, 5.0], [50.0, 0.0])
fit = ERGM.newton_fit(f, zeros(2))
ERGM._separated(f, X, [100.0, 5.0], [50.0, 0.0], fit.θ, fit.se)   # false
```
"""
function _separated(f, X::AbstractMatrix, n_tot::AbstractVector, n_one::AbstractVector,
                    θ::AbstractVector, se::AbstractVector)
    extreme = false
    for r in 1:size(X, 1)
        η = 0.0
        for j in 1:size(X, 2)
            η += X[r, j] * θ[j]
        end
        if (n_one[r] == 0 && η < -18.42) || (n_one[r] == n_tot[r] && η > 18.42)
            extreme = true
            break
        end
    end
    extreme || return false
    any(isnan, se) && return true
    _, grad, hess = f(θ)
    step = try
        -(hess \ grad)
    catch e
        e isa Union{SingularException, LAPACKException, ZeroPivotException,
                    PosDefException} || rethrow()
        return true
    end
    all(isfinite, step) || return true
    return norm(step) > 1e-3 * max(norm(θ), eps())
end

"""
    _warn_separated(context) -> nothing

R's sentence (`mple.existence`, "The MPLE does not exist!") for a separated
design, prefixed with `context` (the fitter's name: `"mple"`, `"cmple"`,
`"ergm_multi"`). `public`: the variants emit the one sentence; a fitter that
runs on [`_mple_fit_design`](@ref) gets it through that function's
`context=` keyword instead.

# Example
```julia
using ERGM
ERGM._warn_separated("my_fit") === nothing   # true: warns "my_fit: the MPLE does not exist …"
```
"""
function _warn_separated(context::AbstractString)
    @warn "$context: the MPLE does not exist (perfect separation): the " *
          "pseudo-likelihood has no finite maximum, and the returned " *
          "coefficients are the point at which Newton stopped on its flat " *
          "asymptote — arbitrarily large, with meaningless standard errors. " *
          "R ergm warns \"The MPLE does not exist!\" for the same design. The " *
          "fit is returned with `converged == false`; a combination of the " *
          "model's statistics perfectly predicts the ties — remove or " *
          "coarsen a term (a nodal covariate, a nodematch/nodemix cell) or " *
          "collect more ties."
    return nothing
end

"""
    _boundary_columns_iterated(X, n_tot, n_one) -> Vector{Tuple{Int,Symbol}}

[`_boundary_columns`](@ref) iterated to the exact limit of the
pseudo-likelihood. Dropping a boundary column restricts the pseudo-likelihood
to the dyads it does not touch, and on THAT design another column can sit at
its boundary (the mixed-sign nodecov example: once `nodecov.x` is fixed at
`-Inf` the three untouched dyads are all ties, so `edges` is at its largest
attainable value). The test is repeated on the reduced design until nothing
more is at a boundary; the result is sorted by column. `public`: this is the
drop test every variant MPLE runs before fitting (ERGMRank, ERGMMulti,
TERGM through [`_mple_fit_design`](@ref)).

# Example
```julia
using ERGM
X = [1.0 0.0; 1.0 2.0]
# Column 2 is at its minimum (its only row is empty); once it is dropped the
# untouched row is all ties, so column 1 is at its maximum on the reduced design
ERGM._boundary_columns(X, [3.0, 2.0], [3.0, 0.0])            # [(2, :min)]
ERGM._boundary_columns_iterated(X, [3.0, 2.0], [3.0, 0.0])   # [(1, :max), (2, :min)]
ERGM._boundary_columns_iterated(X, [3.0, 2.0], [1.0, 1.0])   # empty
```
"""
function _boundary_columns_iterated(X::AbstractMatrix, n_tot::AbstractVector,
                                    n_one::AbstractVector)
    p = size(X, 2)
    dropped = Tuple{Int,Symbol}[]
    rows = collect(1:size(X, 1))
    cols = collect(1:p)
    while !isempty(cols) && !isempty(rows)
        found = _boundary_columns(view(X, rows, cols), view(n_tot, rows), view(n_one, rows))
        isempty(found) && break
        for (jj, side) in found
            push!(dropped, (cols[jj], side))
        end
        gone = Set(cols[jj] for (jj, _) in found)
        rows = [r for r in rows if all(X[r, j] == 0 for j in gone)]
        cols = [j for j in cols if !(j in gone)]
    end
    sort!(dropped; by=first)
    return dropped
end

"""
    _warn_boundary(names, boundary; context, note="R ergm reports the same",
                   noun="dyads") -> nothing

R's sentence (`ergm.checkextreme.model`) for the statistics at the boundary
of their attainable range, one warning per side, prefixed with `context` (the
fitter's name). `boundary` is what [`_boundary_columns_iterated`](@ref)
returns; `names` labels its columns. `note` is the parenthesis after "no
finite maximum pseudo-likelihood estimate exists" — a variant whose R
counterpart does NOT drop (ergm.multi returns a finite asymptote value)
replaces it rather than restating the sentence; `noun` names the rows the
remaining coefficients are fitted on (ERGMRank's are "swap comparisons").
`public`: every variant MPLE emits the one sentence; a fitter that runs on
[`_mple_fit_design`](@ref) gets it through that function's `context=`.

# Example
```julia
using ERGM
ERGM._warn_boundary(["edges", "nodematch.g"], [(2, :min)]; context="mple") === nothing  # true; warns
ERGM._warn_boundary(["edges"], [(1, :max)]; context="fit_ergm_rank",
                    noun="swap comparisons", note="ergm.rank has no drop")
```
"""
function _warn_boundary(names::Vector{String}, boundary::Vector{Tuple{Int,Symbol}};
                        context::AbstractString,
                        note::AbstractString="R ergm reports the same",
                        noun::AbstractString="dyads")
    for (side, word, at) in ((:min, "smallest", "-Inf"), (:max, "largest", "+Inf"))
        cols = [names[j] for (j, s) in boundary if s === side]
        isempty(cols) && continue
        @warn "$context: observed statistic(s) $(join(cols, ", ")) are at their " *
              "$word attainable values. Their coefficients will be fixed at $at " *
              "(no finite maximum pseudo-likelihood estimate exists; $note). " *
              "The remaining coefficients are estimated on the $noun these " *
              "statistics do not touch — the exact limit of the " *
              "pseudo-likelihood — with standard error 0 and p-value 0 recorded " *
              "for the fixed ones."
    end
    return nothing
end

# Core MPLE logistic fit. Returns the pieces mple() and the parametric
# bootstrap need: (coefficients, var_cov, std_errors, loglik, converged,
# n_dyads), with var_cov/std_errors from the inverse observed information.
#
# The likelihood and the optimizer are the ecosystem's shared ones: the
# binomial-row `logistic_derivatives(X, n_tot, n_one)` (each compressed row is
# `n_tot` identical dyads of which `n_one` are ties) maximized by `newton_fit`
# — Newton–Raphson with step halving, converging to `tol` on the objective.
# `newton_fit` already returns NaN standard errors, with a warning, when the
# negative Hessian is not positive definite (a non-identified coefficient), so
# no `try inv(H) catch NaN` is needed here.
#
# A statistic at the boundary of its attainable range (`_boundary_columns`)
# has no finite MPLE: as R ergm does under its default `drop=TRUE`, the
# coefficient is fixed at ∓Inf (standard error 0) and the other coefficients
# are the MPLE on the rows the dropped columns do not touch — the exact limit
# of the pseudo-likelihood as those coefficients go to ∓Inf (their rows'
# probabilities go to the observed 0/1, contributing nothing), and exactly
# what R's `ergm()` returns (edges = logit(12/300), not logit(12/435), on the
# 30-node docs network with a perfectly separated nodematch).
#
# A design on which the pseudo-likelihood has no finite maximum for another
# reason — complete or quasi-complete separation by a combination of columns
# (`_separated`) — is returned with `converged = false` (R: "The MPLE does
# not exist!") instead of the point where Newton met its objective tolerance
# on the flat asymptote: `is_exact` is then false and `mple` warns.
#
# `warn=false` (the parametric bootstrap's refits) silences both warnings:
# a boundary or a separated design in a SIMULATED replicate is not a fact
# about the user's data, and `_mple_bootstrap_cov` reports those replicates
# once, in aggregate.
function _mple_fit(model::ERGMModel; verbose::Bool=false, maxiter::Int=100,
                   tol::Float64=1e-8, warn::Bool=true)
    if verbose
        println("Building design matrix...")
    end
    X, n_tot, n_one = _mple_data(model.network, model.formula.terms, is_directed(model))
    return _mple_fit_design(X, n_tot, n_one, model.formula.terms.names;
                            verbose=verbose, maxiter=maxiter, tol=tol, warn=warn)
end

"""
    _mple_fit_design(X, n_tot, n_one, names; verbose=false, maxiter=100,
                     tol=1e-8, warn=true, context="mple") -> NamedTuple

The pseudo-likelihood fit on a compressed binomial-row design — `X[r, :]` the
change statistics of a class of `n_tot[r]` identical dyads of which
`n_one[r]` are ties — with R ergm's boundary and separation semantics:

1. a statistic at the boundary of its attainable range
   ([`_boundary_columns_iterated`](@ref)) is fixed at ∓Inf (standard error
   0) and the remaining coefficients are the MPLE on the rows the dropped
   columns do not touch — R's `drop=TRUE` — warned by
   [`_warn_boundary`](@ref);
2. otherwise the shared `newton_fit`/`logistic_derivatives` kernel is run
   from zero, and a solution on a flat asymptote ([`_separated`](@ref)) is
   returned with `converged=false`, warned by [`_warn_separated`](@ref).

Returns `(coefficients, var_cov, std_errors, loglik, converged, separated,
n_dyads, n_kept)`: `n_kept` is the number of dyads the finite coefficients
were estimated on — every dyad, or, after a drop, the dyads the dropped
columns do not touch — R's `logLik` nobs attribute, the BIC sample size.
`context` prefixes both warnings (`"cmple"` for TERGM; `warn=false` silences
them, as the bootstrap refits do). This is the entry `mcmle` uses for its
MPLE start (the O(n²) design sweep of `_mple_data` happens once per call),
and `public` so the variants' MPLEs are the same fit rather than a copy.

# Example
```julia
using ERGM
X = [1.0 0.0; 1.0 1.0]          # two dyad classes: edges only / edges + x
r = ERGM._mple_fit_design(X, [100.0, 50.0], [20.0, 25.0], ["edges", "x"]; context="my_fit")
r.converged                     # true
r.coefficients[1] ≈ log(20 / 80)            # true: a saturated design, closed-form logits
r.coefficients[2] ≈ log(25 / 25) - log(20 / 80)   # true
r.n_kept == 150                 # true: nothing dropped
```
"""
function _mple_fit_design(X::Matrix{Float64}, n_tot::Vector{Float64},
                          n_one::Vector{Float64}, names::Vector{String};
                          verbose::Bool=false, maxiter::Int=100, tol::Float64=1e-8,
                          warn::Bool=true, context::AbstractString="mple")
    p = size(X, 2)
    n_dyads = sum(n_tot)

    boundary = _boundary_columns_iterated(X, n_tot, n_one)
    if !isempty(boundary)
        warn && _warn_boundary(names, boundary; context=context)
        return _mple_fit_dropped(X, n_tot, n_one, boundary, p, n_dyads;
                                 verbose=verbose, maxiter=maxiter, tol=tol, warn=warn,
                                 context=context)
    end

    if verbose
        println("Fitting logistic regression ($(size(X, 1)) unique rows, $(Int(n_dyads)) dyads)...")
    end

    f = logistic_derivatives(X, n_tot, n_one)
    fit = newton_fit(f, zeros(p); maxiter=maxiter, tol=tol)
    separated = _separated(f, X, n_tot, n_one, fit.θ, fit.se)
    separated && warn && _warn_separated(context)

    return (coefficients=fit.θ, var_cov=fit.vcov, std_errors=fit.se,
            loglik=fit.loglik, converged=fit.converged && !separated,
            separated=separated, n_dyads=n_dyads, n_kept=n_dyads)
end

# The reduced fit behind a boundary statistic (see `_mple_fit`): columns
# `dropped` fixed at ∓Inf, the rest fit on the rows where every dropped
# column is zero, then padded back to the full parameter vector.
function _mple_fit_dropped(X, n_tot, n_one, boundary, p::Int, n_dyads;
                           verbose::Bool, maxiter::Int, tol::Float64, warn::Bool=true,
                           context::AbstractString="mple")
    dropped = Dict(boundary)
    keep_cols = [j for j in 1:p if !haskey(dropped, j)]
    keep_rows = [r for r in 1:size(X, 1) if all(X[r, j] == 0 for j in keys(dropped))]
    fixed(j) = dropped[j] === :min ? -Inf : Inf

    θ = zeros(p); se = zeros(p); V = zeros(p, p)
    loglik = 0.0
    converged = true
    separated = false
    # R's `logLik` nobs attribute: the dyads the dropped columns do not touch
    # (300 of 435 on the docs' separated network) — the BIC sample size
    n_kept = sum(n_tot[keep_rows]; init=0.0)
    if !isempty(keep_cols)
        Xr = X[keep_rows, keep_cols]
        verbose && println("Fitting logistic regression ($(size(Xr, 1)) unique rows, " *
                           "$(Int(n_kept)) dyads; $(length(dropped)) " *
                           "coefficient(s) fixed at ±Inf)...")
        f = logistic_derivatives(Xr, n_tot[keep_rows], n_one[keep_rows])
        fit = newton_fit(f, zeros(length(keep_cols)); maxiter=maxiter, tol=tol)
        θ[keep_cols] = fit.θ
        se[keep_cols] = fit.se
        V[keep_cols, keep_cols] = fit.vcov
        loglik = fit.loglik
        separated = _separated(f, Xr, n_tot[keep_rows], n_one[keep_rows], fit.θ, fit.se)
        separated && warn && _warn_separated(context)
        converged = fit.converged && !separated
    end
    for j in keys(dropped)
        θ[j] = fixed(j)
    end
    return (coefficients=θ, var_cov=V, std_errors=se, loglik=loglik,
            converged=converged, separated=separated, n_dyads=n_dyads, n_kept=n_kept)
end

"""
    mple(model::ERGMModel; verbose=false, se=:hessian, n_boot=100,
         boot_burnin=nothing, boot_interval=nothing, maxiter=100, tol=1e-8,
         rng=Random.default_rng()) -> ERGMResult

Fit an ERGM using Maximum Pseudo-Likelihood Estimation: a logistic regression
of the dyad indicators on the change statistics, solved by Newton–Raphson on
the ecosystem's shared kernel (`Networks.logistic_derivatives` +
`Networks.newton_fit`) to a tolerance of `tol` on the pseudo-log-likelihood.
On the Florentine marriage network the dyad-independent fit agrees with R
`ergm` to 1e-12 on the coefficients and 5e-8 on the standard errors
(provenanced fixture, asserted at 1e-6).

# Arguments
- `model::ERGMModel`: The ERGM model specification
- `verbose::Bool=false`: Print progress information
- `maxiter::Int=100`, `tol::Float64=1e-8`: Newton iteration cap and
  convergence tolerance (see `Networks.newton_fit`). A fit that exhausts
  `maxiter` is returned with `converged = false` and a warning
- `se::Symbol=:hessian`: How to compute standard errors:
  - `:hessian` — inverse of the observed pseudo-likelihood information.
    **Caution:** for models with dyad-dependent terms these are typically
    *anticonservative* (too small), because the pseudo-likelihood treats
    dependent dyads as independent observations; the resulting p-values
    are then too optimistic. For dyad-independent models the
    pseudo-likelihood is the true likelihood and these SEs are correct.
  - `:bootstrap` — parametric bootstrap: simulate `n_boot` networks from
    the model at the MPLE estimate (via the MCMC sampler), refit the MPLE
    on each, and use the empirical covariance of the refitted coefficients.
    Honest under dyad dependence (though the MPLE point estimate itself may
    still be biased; consider `method=:mcmle`). A replicate on which the
    MPLE does not exist — a simulated network with no triangle under
    `Triangle()`, no shared-partner edge under `GWESP`, or a separated
    design — is **excluded** from the covariance (its refit is `-Inf`/NaN,
    which would make every standard error NaN): ONE warning names how many
    of `n_boot` were excluded, `fit.boot_replicates` holds every refit (the
    excluded ones as NaN rows) and `approximations(fit)` records it. At
    least two finite refits are required (`ArgumentError` otherwise). A
    point estimate that itself carries a `±Inf` coefficient (see below)
    cannot be simulated from, so `se=:bootstrap` is refused for it.
- `n_boot::Int=100`: Number of bootstrap replicates (only for `se=:bootstrap`)
- `boot_burnin`, `boot_interval`: MCMC controls for the bootstrap
  simulations; default to the dyad-scaled `mcmle` defaults
  (`20 * n_dyads` and `max(100, n_dyads ÷ 10)`)
- `rng::AbstractRNG=Random.default_rng()`: Source of the bootstrap
  simulation randomness (reproducible seeding)

# Returns
- `ERGMResult`: Fitted model results (`se_type` records `:hessian` or
  `:bootstrap`)

Note: the reported log-likelihood is the maximized *pseudo*-log-likelihood;
AIC/BIC derived from it are only heuristics for dependence models.
`dof(fit)` counts the *finite* coefficients and the BIC sample size is the
number of dyads they were estimated on (R's `logLik` df and nobs
attributes), so after a drop (below) AIC/BIC are R's numbers too.

**Perfect separation** that `_boundary_columns` cannot see — a combination
of statistics that predicts every tie — leaves the pseudo-likelihood
without a maximum. R warns "The MPLE does not exist!"; `mple` detects the
asymptote (a perfectly predicted dyad at a fitted probability within 1e-8 of
0/1 while Newton is still moving) and returns the fit with
`converged == false`, a warning, and the caveat in `show` and
`approximations`; `is_exact(fit)` is then `false`.

**A statistic at the boundary of its attainable range** — a `NodeMatch`
with no within-group tie, a `Triangle` on a triangle-free network, … — has
no finite MPLE. As R ergm does (its default `drop=TRUE`), `mple` warns
"observed statistic(s) … are at their smallest attainable values. Their
coefficients will be fixed at -Inf", returns that coefficient as `-Inf`
(`+Inf` at the largest value) with standard error 0 and p-value 0, and
fits the remaining coefficients on the dyads the dropped statistic does not
touch — the exact limit of the pseudo-likelihood, and R's numbers.
`Networks.approximations(fit)` lists the fixed coefficients; `show` prints
a note under the table. `mcmle` refuses such a model instead (see there).

Dyads masked as missing (`Networks.set_missing_dyad!`) are excluded from the
pseudo-likelihood: an unobserved tie status is not a response, so those
dyads contribute no logistic-regression row and `nobs` decreases
accordingly. Their face values still condition the change statistics of the
observed dyads. This is the standard *available-case* pseudo-likelihood — a
principled treatment of missing data — so `mple` declares
`Networks.supports_missing(mple) == true` and needs no `missing` keyword;
the fit records `missing_method = :available_case` (`:none` when nothing is
masked).

**Caveat for `se=:bootstrap` on a masked network**: the bootstrap *simulates*
from the fitted model, and a simulated network has a value at every dyad,
so the MH sampler conditions the masked dyads on their face value. The
point estimate is available-case, but the bootstrap SEs around it are not;
a warning is emitted. For maximum likelihood under missing data use
`mcmle(model; missing=:mle)`.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
fit = mple(ERGMModel(ERGMFormula([Edges(), NodeCov(:wealth)]), net))
fit.converged            # true
round.(coef(fit); digits=3)   # [-2.595, 0.011]  (R ergm: -2.594929, 0.010546)
```
"""
function mple(model::ERGMModel{T,D}; verbose::Bool=false,
              se::Symbol=:hessian,
              n_boot::Int=100,
              boot_burnin::Union{Nothing,Int}=nothing,
              boot_interval::Union{Nothing,Int}=nothing,
              maxiter::Int=100,
              tol::Float64=1e-8,
              rng::AbstractRNG=Random.default_rng()) where {T,D}
    check_se(se, (:hessian, :bootstrap); context="mple")

    fit = _mple_fit(model; verbose=verbose, maxiter=maxiter, tol=tol)
    coefficients = fit.coefficients
    p = length(coefficients)

    # An unconverged fit is a loud result in this ecosystem, never a silent
    # one: it is still returned (converged=false is recorded), but said. A
    # separated design has already been warned about by `_mple_fit` (R's
    # "The MPLE does not exist!"); the Newton cap is the other way to get
    # here.
    fit.converged || fit.separated || @warn "mple: the Newton iteration did not " *
        "converge within maxiter=$maxiter (the pseudo-likelihood may be " *
        "unbounded — perfect separation or a degenerate statistic). The " *
        "returned coefficients are the last iterate and `converged == false`; " *
        "check the model for a statistic that is constant or perfectly " *
        "predicts the ties."

    var_cov, std_errors = fit.var_cov, fit.std_errors
    boot_replicates = nothing
    if se == :bootstrap
        # A coefficient fixed at ∓Inf cannot be simulated from (the sampler's
        # θ'δ would be NaN wherever the dropped statistic changes), so there
        # is nothing to bootstrap: say so instead of returning NaN errors.
        if any(isinf, coefficients)
            fixed = [terms_name for (terms_name, c) in
                     zip(model.formula.terms.names, coefficients) if isinf(c)]
            throw(ArgumentError(
                "mple: se=:bootstrap is not available when a coefficient is fixed " *
                "at ±Inf by a statistic at the boundary of its attainable range " *
                "($(join(fixed, ", "))): a network cannot be simulated at an " *
                "infinite coefficient. Remove the term (as R's drop=TRUE does) " *
                "or keep the default se=:hessian, which reports standard error 0 " *
                "for the fixed coefficient and the inverse-Hessian errors of the rest."))
        end
        var_cov, std_errors, boot_replicates =
            _mple_bootstrap_cov(model, coefficients; n_boot=n_boot,
                                boot_burnin=boot_burnin, boot_interval=boot_interval,
                                rng=rng, verbose=verbose)
    end

    # Z-values and p-values (a coefficient fixed at ∓Inf by a boundary
    # statistic has SE 0: z = ∓Inf and p = 0, as R prints them)
    z_values = coefficients ./ std_errors
    p_values = z_pvalues(z_values)
    for k in eachindex(coefficients)
        isinf(coefficients[k]) && (p_values[k] = 0.0)
    end

    # AIC and BIC (based on the pseudo-likelihood). The degrees of freedom
    # are the FINITE coefficients and the BIC sample size the dyads they were
    # estimated on — every dyad, or, after R's drop of a boundary statistic,
    # the dyads the dropped term does not touch (`fit.n_kept`): R's
    # `logLik.ergm` df and nobs attributes (df = 1, nobs = 300 of 435 on the
    # docs' separated network; `nobs(fit)` itself stays every observed dyad).
    k = count(isfinite, coefficients)
    aic = -2 * fit.loglik + 2 * k
    bic = -2 * fit.loglik + k * log(fit.n_kept)

    return ERGMResult(
        model,
        coefficients,
        std_errors,
        z_values,
        p_values,
        var_cov,
        fit.loglik,
        aic,
        bic,
        :mple,
        fit.converged,
        nothing,
        se,
        n_missing_dyads(model.network) == 0 ? :none : :available_case,
        var_cov,          # no MCMC sample: the Fisher part is the whole covariance
        zeros(p),         # ... and there is no Monte-Carlo component
        nothing,
        Int[],            # no chains
        boot_replicates
    )
end

# MPLE is the ecosystem's declared missing-data-capable ERGM estimator: it
# drops masked dyads from the pseudo-likelihood (available-case), which is a
# principled treatment, not a face-value read.
supports_missing(::typeof(mple)) = true

# Parametric-bootstrap covariance of the MPLE: simulate n_boot networks at
# θ̂, refit the MPLE on each, and return the empirical covariance and SEs of
# the refitted coefficients.
#
# The loop itself is `Networks.bootstrap_cov` — the ONE shared bootstrap of the
# ecosystem (Networks.jl `src/bootstrap.jl`), which the count/rank/multilayer
# MPLEs and REM's repeated control sampling also call. This function supplies
# only the two callbacks that are ERGM's: how to simulate at θ̂, and how to refit.
function _mple_bootstrap_cov(model::ERGMModel, θ̂::Vector{Float64};
                             n_boot::Int, boot_burnin, boot_interval,
                             rng::AbstractRNG, verbose::Bool)
    # The same dyad-scaled defaults as every sampler (`_mcmc_defaults`)
    burnin, interval = _resolve_mcmc_controls(model, boot_burnin, boot_interval)

    if verbose
        println("Parametric bootstrap: simulating $n_boot networks at the MPLE...")
    end

    # The MPLE point estimate is available-case, but the bootstrap has to
    # *simulate*, and the sampler can only condition masked dyads on their
    # face value. Say so, and opt in explicitly rather than tripping the
    # sampler's own guard.
    _warn_condition_on_face(model.network,
                            "the MPLE parametric bootstrap (`se=:bootstrap`)")

    simulate(rng, B) = sample_networks(model, θ̂; n_sim=B, burnin=burnin,
                                       interval=interval, rng=rng,
                                       missing=:condition_on_face)

    # Same (already materialized) formula, simulated network. The simulated
    # networks carry the observed network's attributes, so re-validation and
    # materialization are no-ops. A replicate on which the MPLE does not
    # exist — a simulated network with no triangle under `Triangle()`, no
    # shared partner under `GWESP`, … (a boundary statistic, coefficient
    # ∓Inf), or a separated design — has no finite refit: its row is NaN,
    # excluded from the covariance below and counted. The refit is silent
    # (`warn=false`): R's boundary sentence would otherwise fire once per
    # replicate about a SIMULATED network, and `bootstrap_cov` runs the
    # refits on every thread.
    function refit(sim)
        r = _mple_fit(ERGMModel(model.formula, sim; reference=model.reference);
                      warn=false)
        return r.converged ? r.coefficients : fill(NaN, length(θ̂))
    end

    boot = bootstrap_cov(refit, simulate, θ̂; n_boot=n_boot, rng=rng)
    replicates = boot.replicates
    ok = [all(isfinite, view(replicates, b, :)) for b in 1:n_boot]
    n_ok = count(ok)
    n_ok == n_boot && return boot.vcov, boot.se, replicates

    n_ok >= 2 || throw(ArgumentError(
        "mple: se=:bootstrap — only $n_ok of the $n_boot bootstrap refits had a " *
        "finite MPLE (the others simulated a network on which a statistic sits " *
        "at the boundary of its attainable range, or is perfectly separated); a " *
        "covariance needs at least 2. The model is near-degenerate at its " *
        "MPLE: increase n_boot, simplify the formula, or refit with method=:mcmle."))
    @warn "mple: se=:bootstrap — $(n_boot - n_ok) of the $n_boot bootstrap refits " *
          "had no finite MPLE (the simulated network put a statistic at the " *
          "boundary of its attainable range — e.g. no triangle, no shared " *
          "partner — or perfectly separated the ties) and were excluded; the " *
          "standard errors are the empirical covariance of the $n_ok finite " *
          "refits. This is about the simulated replicates, not about the observed " *
          "network. `fit.boot_replicates` holds every refit (NaN rows excluded); " *
          "`approximations(fit)` records the exclusion."
    V = Matrix{Float64}(cov(replicates[ok, :]))
    return V, sqrt.(max.(diag(V), 0.0)), replicates
end

"""
    fit_ergm(net, terms; method=:mple, kwargs...) -> ERGMResult
    ergm(net, terms; method=:mple, kwargs...) -> ERGMResult

Fit an ERGM to the observed network `net` with the model `terms` — R ergm's
`ergm(net ~ edges + triangle + ...)`. `ergm` is a `const` alias of `fit_ergm`
(`ergm === fit_ergm`): the statnet name and the ecosystem's `fit_<model>`
name are one function.

Builds `ERGMModel(ERGMFormula(terms), net)` (validating every term against
the network — a missing attribute, a directed-only term on an undirected
network, or a two-mode network throws an `ArgumentError` before any fitting)
and dispatches on `method`:

- `method=:mple` (default) — maximum pseudo-likelihood, [`mple`](@ref).
  Exact for dyad-independent formulas; fast but approximate under dyadic
  dependence (see the caveat `show` prints).
- `method=:mcmle` — Monte-Carlo maximum likelihood, [`mcmle`](@ref).

Every other keyword is forwarded unchanged to the chosen estimator:

| keyword | estimator | meaning |
|:--|:--|:--|
| `verbose` | both | progress output |
| `se`, `n_boot`, `boot_burnin`, `boot_interval` | `mple` | `:hessian` or `:bootstrap` standard errors and the bootstrap's controls (replicates without a finite MPLE are excluded, warned once) |
| `maxiter`, `tol` | `mple` | Newton iteration cap / tolerance of the logistic fit |
| `n_samples`, `burnin`, `interval`, `maxiter` | `mcmle` | MCMC sample size per iteration, dyad-scaled burn-in / thinning, iteration cap |
| `conv_threshold`, `hotelling_alpha`, `gamma0`, `max_step_norm` | `mcmle` | convergence tests and Hummel step control |
| `bridge_rungs`, `bridge_samples` | `mcmle` | path-sampling log-likelihood controls |
| `missing` | `mcmle` | `:error` (default), `:mle` (missing-data maximum likelihood) or `:condition_on_face` for masked dyads |
| `obs_burnin`, `obs_interval` | `mcmle` | controls of the constrained chain under `missing=:mle` |
| `rng` | both | the `AbstractRNG` every Monte-Carlo draw flows from |

# Arguments
- `net::Network`: the observed (one-mode) network
- `terms`: a single term, a vector of terms, or a vector mixing terms with
  vectors of terms — `[Edges(), Degree(0:2)]` is spliced into one term per
  degree, as statnet's `degree(0:2)`. Anything else is an `ArgumentError`
  naming the offending element; swapping the two positional arguments
  (`fit_ergm(terms, net)`) is an `ArgumentError` too, not a `MethodError`.
- `method::Symbol=:mple`: `:mple` or `:mcmle`

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
fit = ergm(net, [Edges(), NodeCov(:wealth)])            # MPLE (exact here)
round.(coef(fit); digits=3)                              # [-2.595, 0.011]
fit_dep = fit_ergm(net, [Edges(), GWESP(0.5)]; method=:mcmle,
                   n_samples=500, rng=Xoshiro(1))
fit_dep.converged                                        # true
```
"""
function fit_ergm(net::Network, terms::AbstractVector; method::Symbol=:mple, kwargs...)
    formula = ERGMFormula(_collect_terms(terms))
    model = ERGMModel(formula, net)

    if method === :mple
        return mple(model; kwargs...)
    elseif method === :mcmle
        return mcmle(model; kwargs...)
    else
        throw(ArgumentError("unknown estimation method $(repr(method)); expected " *
                            ":mple (maximum pseudo-likelihood) or :mcmle " *
                            "(Monte-Carlo maximum likelihood)"))
    end
end

# A single term needs no brackets
fit_ergm(net::Network, term::AbstractERGMTerm; kwargs...) =
    fit_ergm(net, AbstractERGMTerm[term]; kwargs...)

# Swapped positional arguments: say so, instead of a raw MethodError
const _SWAPPED_ARGS_MSG =
    "arguments are swapped: call fit_ergm(net, terms) — the network first, " *
    "then the term or vector of terms (R's `ergm(net ~ terms)` order)"
fit_ergm(terms::AbstractVector, net::Network; kwargs...) =
    throw(ArgumentError(_SWAPPED_ARGS_MSG))
fit_ergm(term::AbstractERGMTerm, net::Network; kwargs...) =
    throw(ArgumentError(_SWAPPED_ARGS_MSG))

# Two-mode wrappers are refused with the same message as a flagged Network
fit_ergm(net::BipartiteNetwork, terms; kwargs...) = _refuse_two_mode(net)

"""
    ergm(net, terms; method=:mple, kwargs...) -> ERGMResult

The statnet-style name of [`fit_ergm`](@ref) — a `const` alias, so
`ergm === fit_ergm` and the two names share one method table, one docstring
of keywords and one set of error messages. See `fit_ergm` for the full
keyword vocabulary.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
ergm(net, [Edges(), NodeCov(:wealth)]) === nothing   # false — a fitted ERGMResult
ergm === fit_ergm                                     # true
```
"""
const ergm = fit_ergm
