"""
Maximum Pseudo-Likelihood Estimation (MPLE) for ERGMs.

MPLE treats each potential edge as an independent observation and fits
a logistic regression of the dyad indicators on the add-direction change
statistics, each computed conditional on the rest of the observed network.
This is fast but may be biased for models with strong dependencies.

The logistic fit itself is the ecosystem's shared kernel: the binomial-row
form of `NetworkCore.logistic_derivatives` maximized by `NetworkCore.newton_fit`.
ERGM.jl adds only the design construction.
"""

# Numerically stable log(1 + exp(x)) — used by `_dyad_independent_logZ`
_log1pexp(x::Float64) = x > 35 ? x : (x < -35 ? exp(x) : log1p(exp(x)))

"""
    _mple_data(net, terms::TermSet, directed::Bool)

Build compressed MPLE data: unique change-statistic rows with the number of
dyads (`n_tot`) and the number of observed edges (`n_one`) sharing each row.
Compressing identical rows keeps memory O(unique rows) instead of O(n²).

Dyads masked as missing (see `NetworkCore.set_missing_dyad!`) are excluded: their
tie status is unobserved, so they contribute no response row. Their face
values still enter the *predictors* of the remaining dyads, since change
statistics are computed conditional on the rest of the observed network.

The row table is keyed on an `NTuple{p,Float64}` — the change-statistic
tuple built term by term by the generated `_change_stat_tuple` lives on the
stack for ANY `p` (a `Base.map` over the term tuple boxed every value from 32
terms on) — so the sweep allocates O(unique rows) (the Dict and the returned
arrays), not a fresh `Vector{Float64}` per dyad.
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
    _boundary_columns_once(X, n_tot, n_one) -> Vector{Tuple{Int,Symbol}}

The design columns whose observed statistic sits at the boundary of its
attainable range, read off the compressed pseudo-likelihood design — R
ergm's attainable-range test (`ergm.checkextreme.model`). Under the
pseudo-likelihood each dyad `r` contributes `X[r, j]` to statistic `j` if
tied, so the smallest value the statistic can attain is `Σ_r n_tot[r] ·
min(0, X[r, j])` and the observed value sits AT it iff every dyad with
`X[r, j] > 0` is a non-tie (`n_one[r] == 0`) and every dyad with
`X[r, j] < 0` is a tie (`n_one[r] == n_tot[r]`); the mirror condition is the
largest attainable value. Internal: the variants run the iterated form,
[`boundary_columns`](@ref ERGM.Extension.boundary_columns). The sign pattern of the column is irrelevant: a
`nodecov` whose change statistic `x_i + x_j` takes both signs is at its
minimum whenever the positive-change dyads are all empty and the
negative-change dyads all tied (such a column used to be
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
ERGM._boundary_columns_once(X, [3.0, 2.0], [0.0, 2.0])   # [(1, :min)]: +1 dyads empty, −1 dyads tied
ERGM._boundary_columns_once(X, [3.0, 2.0], [3.0, 0.0])   # [(1, :max)]
ERGM._boundary_columns_once(X, [3.0, 2.0], [1.0, 1.0])   # empty: strictly inside its range
```
"""
function _boundary_columns_once(X::AbstractMatrix, n_tot::AbstractVector, n_one::AbstractVector)
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

# R ergm's own sentences for the same design, which close ERGM.jl's warnings
# (`mple_fit_design`'s `note` keyword replaces them for a package whose R
# counterpart behaves differently)
const _R_MPLE_NOT_EXIST = "R ergm warns \"The MPLE does not exist!\" for the same design."
const _R_BOUNDARY_NOTE = "R ergm reports the same"
const _R_NOT_VARYING_NOTE = "R ergm warns \"Model statistics ... are not varying\" and reports NA"
const _R_LINEAR_DEPENDENCE_NOTE = "R's glm reports NA; R ergm warns \"linear dependence ... the " *
                                  "model is nonidentifiable\""

"""
    boundary_columns(X, n_tot, n_one; fixed=[]) -> Vector{Tuple{Int,Symbol}}

The columns of a compressed pseudo-likelihood design whose statistic is at
the boundary of its attainable range, as `(column, :min)` or
`(column, :max)` sorted by column — R ergm's drop test, iterated to the
exact limit of the pseudo-likelihood. `X[r, :]` holds the change statistics
of a class of `n_tot[r]` identical dyads (or comparisons) of which `n_one[r]`
are ties.

A column is at its minimum when every row with a positive change statistic
is a non-tie and every row with a negative one is a tie (the mirror for the
maximum); the sign pattern of the column does not matter. Its
pseudo-log-likelihood then increases monotonically as the coefficient goes
to `-Inf` (`:min`) or `+Inf` (`:max`), so no finite estimate exists. A
perfectly separated `nodematch` (no within-group tie) is the textbook case.

Dropping a boundary column restricts the pseudo-likelihood to the rows it
does not touch, and on that design another column can sit at its boundary
(a mixed-sign `nodecov`: once it is fixed at `-Inf` the untouched dyads are
all ties, so `edges` is at its largest value). The test is therefore
repeated on the reduced design until nothing more is at a boundary.

`fixed` seeds the iteration with columns known to be at a boundary
beforehand, normally [`extreme_statistics`](@ref ERGM.Extension.extreme_statistics):
a statistic whose observed value equals an end of its
[`attainable_range`](@ref ERGM.Extension.attainable_range). The design cannot
show such a column when its change statistics are all zero (a `Triangle` on
a network with no two-path). Seeds are included in the result.

Part of [`ERGM.Extension`](@ref): every ERGM-family pseudo-likelihood runs
this test, directly or through
[`mple_fit_design`](@ref ERGM.Extension.mple_fit_design).

# Example
```julia
using ERGM
X = [1.0 0.0; 1.0 2.0]
# Column 2 is at its minimum (its only row is empty); once it is dropped the
# untouched row is all ties, so column 1 is at its maximum on the reduced design
ERGM.Extension.boundary_columns(X, [3.0, 2.0], [3.0, 0.0])   # [(1, :max), (2, :min)]
ERGM.Extension.boundary_columns(X, [3.0, 2.0], [1.0, 1.0])   # empty
# A column of zeros sits at no design boundary; seeded as one, it is listed
Z = [1.0 0.0; 1.0 0.0]
ERGM.Extension.boundary_columns(Z, [3.0, 2.0], [1.0, 1.0]; fixed=[(2, :min)])  # [(2, :min)]
```
"""
function boundary_columns(X::AbstractMatrix, n_tot::AbstractVector,
                          n_one::AbstractVector;
                          fixed::AbstractVector{Tuple{Int,Symbol}}=Tuple{Int,Symbol}[])
    p = size(X, 2)
    dropped = Tuple{Int,Symbol}[fixed...]
    gone0 = Set(j for (j, _) in fixed)
    rows = [r for r in 1:size(X, 1) if all(X[r, j] == 0 for j in gone0)]
    cols = [j for j in 1:p if !(j in gone0)]
    while !isempty(cols) && !isempty(rows)
        found = _boundary_columns_once(view(X, rows, cols), view(n_tot, rows), view(n_one, rows))
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
    warn_boundary(names, boundary; context, note="R ergm reports the same",
                  noun="dyads") -> nothing

R ergm's warning (`ergm.checkextreme.model`) for the statistics at the
boundary of their attainable range, one warning per side, prefixed with
`context` (the fitter's name). `boundary` is what
[`boundary_columns`](@ref ERGM.Extension.boundary_columns) returns, and
`names` labels its columns.

- `note` is the parenthesis after "no finite maximum pseudo-likelihood
  estimate exists". A package whose R counterpart does not drop (ergm.multi
  returns a finite value) says so here instead of restating the sentence.
- `noun` names the rows the remaining coefficients are fitted on (ERGMRank's
  are "swap comparisons").

Part of [`ERGM.Extension`](@ref): every ERGM-family estimator emits this one
sentence. A fitter that runs on
[`mple_fit_design`](@ref ERGM.Extension.mple_fit_design) gets it from that
function, worded through its `context`, `noun` and `note` keywords.

# Example
```julia
using ERGM
ERGM.Extension.warn_boundary(["edges", "nodematch.g"], [(2, :min)]; context="mple") === nothing  # true; warns
ERGM.Extension.warn_boundary(["edges"], [(1, :max)]; context="fit_ergm_rank",
                             noun="swap comparisons", note="ergm.rank has no drop")
```
"""
function warn_boundary(names::Vector{String}, boundary::Vector{Tuple{Int,Symbol}};
                       context::AbstractString,
                       note::AbstractString=_R_BOUNDARY_NOTE,
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

# The pseudo-likelihood design of a model with `Offset` terms: the estimated
# columns `free`, the rows `keep` that remain, and the fixed offset
# contribution `η` of every row. An infinite offset fixes the dyads its
# statistic reaches: where the offset's contribution to the tie's log-odds is
# −∞ the tie is FORBIDDEN (a structural zero), where it is +∞ the tie is
# FORCED. Those dyads have probability 1 of being as the constraint says and
# are dropped, contributing nothing. An observed network that violates a
# constraint (a tie on a forbidden dyad, no tie on a forced one) has
# probability 0 under the model and is refused.
function _offset_design(X::AbstractMatrix, n_tot::AbstractVector, n_one::AbstractVector,
                        names::Vector{String}, mask::AbstractVector{Bool},
                        vals::AbstractVector{<:Real})
    free = findall(!, mask)
    off = findall(mask)
    keep = trues(size(X, 1))
    η = zeros(size(X, 1))
    for r in 1:size(X, 1)
        fixed = 0                      # −1 forbidden, +1 forced
        for k in off
            x = X[r, k]
            x == 0 && continue
            c = vals[k]
            if isinf(c)
                side = (c > 0) == (x > 0) ? 1 : -1
                fixed != 0 && fixed != side && throw(ArgumentError(
                    "infinite offsets contradict each other: on some dyads " *
                    "$(names[k]) $(side == 1 ? "forces" : "forbids") a tie that " *
                    "another infinite offset $(side == 1 ? "forbids" : "forces")"))
                fixed = side
                violated = side == -1 ? n_one[r] > 0 : n_one[r] < n_tot[r]
                violated && throw(ArgumentError(
                    "the observed network " *
                    (side == -1 ? "has a tie on a dyad where $(names[k]) = $c forbids one " *
                                  "(a structural zero)" :
                                  "lacks a tie on a dyad where $(names[k]) = $c forces one") *
                    ": the model gives the observed network probability 0. Remove " *
                    "the offset, or make the network satisfy the constraint."))
                keep[r] = false
            else
                η[r] += c * x
            end
        end
    end
    return free, keep, η
end

# The pseudo-likelihood fit of a design whose model may carry offsets: the
# offset columns enter as the fixed linear-predictor offset of the shared
# kernel (`logistic_derivatives(...; offset)`), the estimated ones are fitted
# by `mple_fit_design`, and the result is reassembled on all `p`
# coordinates — the fixed values at the offsets, standard error 0 and zero
# covariance there (R's report of an offset term).
function _mple_fit_offsets(X::Matrix{Float64}, n_tot::Vector{Float64},
                           n_one::Vector{Float64}, names::Vector{String},
                           mask::AbstractVector{Bool}, vals::AbstractVector{<:Real};
                           extreme::AbstractVector{Tuple{Int,Symbol}}=Tuple{Int,Symbol}[],
                           kwargs...)
    any(mask) || return mple_fit_design(X, n_tot, n_one, names; extreme=extreme, kwargs...)
    p = size(X, 2)
    free, keep, η = _offset_design(X, n_tot, n_one, names, mask, vals)
    # the extreme columns, renumbered among the estimated ones
    extreme = Tuple{Int,Symbol}[(findfirst(==(j), free)::Int, side) for (j, side) in extreme if !mask[j]]
    Xk, tk, ok, ηk = X[keep, free], n_tot[keep], n_one[keep], η[keep]
    θ = Float64.(vals)
    se = zeros(p)
    V = zeros(p, p)
    if isempty(free)
        # Every coefficient fixed: nothing to fit, the pseudo-log-likelihood
        # is evaluated at the offsets
        loglik = sum(ok[r] * ηk[r] - tk[r] * _log1pexp(ηk[r]) for r in eachindex(ηk); init=0.0)
        return (coefficients=θ, var_cov=V, std_errors=se, loglik=loglik, converged=true,
                separated=false, separated_terms=String[], n_dyads=sum(n_tot),
                n_kept=sum(tk))
    end
    r = mple_fit_design(Xk, tk, ok, names[free]; offset=ηk, extreme=extreme, kwargs...)
    θ[free] = r.coefficients
    se[free] = r.std_errors
    V[free, free] = r.var_cov
    return (coefficients=θ, var_cov=V, std_errors=se, loglik=r.loglik,
            converged=r.converged, separated=r.separated,
            separated_terms=r.separated_terms, n_dyads=sum(n_tot), n_kept=r.n_kept)
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
# A statistic at the boundary of its attainable range (`_boundary_columns_once`)
# has no finite MPLE: as R ergm does under its default `drop=TRUE`, the
# coefficient is fixed at ∓Inf (standard error 0) and the other coefficients
# are the MPLE on the rows the dropped columns do not touch — the exact limit
# of the pseudo-likelihood as those coefficients go to ∓Inf (their rows'
# probabilities go to the observed 0/1, contributing nothing), and exactly
# what R's `ergm()` returns (edges = logit(12/300), not logit(12/435), on the
# 30-node docs network with a perfectly separated nodematch).
#
# A design on which the pseudo-likelihood has no finite maximum for another
# reason — complete or quasi-complete separation by a combination of columns,
# decided by the shared `NetworkCore.logistic_separation` verdict (an exact
# linear programme, as R's `mple.existence`) — follows the ecosystem's
# separation policy: warned (R: "The MPLE does not exist!"), returned with
# `converged = false` and the separated terms flagged, and `mple` withholds
# z values, p-values and intervals; `is_exact` is then false.
#
# `warn=false` (the parametric bootstrap's refits) silences both warnings:
# a boundary or a separated design in a SIMULATED replicate is not a fact
# about the user's data, and `_mple_bootstrap_cov` reports those replicates
# once, in aggregate.
function _mple_fit(model::ERGMModel; verbose::Bool=false, maxiter::Int=100,
                   tol::Float64=1e-8, warn::Bool=true, drop::Bool=true)
    if verbose
        println("Building design matrix...")
    end
    X, n_tot, n_one = _mple_data(model.network, model.formula.terms, is_directed(model))
    mask, vals = _offset_info(model)
    extreme = extreme_statistics(model.formula.terms, model.network)
    drop || _refuse_no_drop(model, (X, n_tot, n_one), extreme; context="mple")
    return _mple_fit_offsets(X, n_tot, n_one, model.formula.terms.names, mask, vals;
                             extreme=extreme, verbose=verbose, maxiter=maxiter, tol=tol,
                             warn=warn)
end

"""
    mple_fit_design(X, n_tot, n_one, names; context="mple", estimate="MPLE",
                    noun="dyads", note=(;), warn=true, extreme=[], offset=nothing,
                    maxiter=100, tol=1e-8, verbose=false) -> NamedTuple

The pseudo-likelihood fit of a compressed binomial-row design, with R ergm's
boundary and separation semantics. Row `r` of `X` holds the change
statistics of a class of `n_tot[r]` identical dyads (or comparisons), of
which `n_one[r]` are ties; `names` labels the columns. The steps are:

1. **Drop.** A statistic at the boundary of its attainable range
   ([`boundary_columns`](@ref ERGM.Extension.boundary_columns), seeded with
   `extreme`) is fixed at ∓Inf with standard error 0. The other coefficients
   are fitted on the rows the dropped columns do not touch, which is R's
   `drop=TRUE` and the exact limit of the pseudo-likelihood. Pass the
   statistics whose observed value is at an end of their range as `extreme`
   ([`extreme_statistics`](@ref ERGM.Extension.extreme_statistics)): the
   design cannot show them when their change statistics are all zero.
2. **Aliasing.** A statistic that does not vary on the rows fitted, or is a
   linear combination of the statistics before it, has no identifiable
   coefficient. It is reported as `NaN` (standard error `NaN`; R's `glm`
   gives `NA`) and the rest is fitted without it.
3. **Newton and separation.** NetworkCore's `newton_fit` on
   `logistic_derivatives` fits the remaining columns from zero (or from a
   continuation start when there is an `offset`). NetworkCore's
   `logistic_separation` verdict is computed on the design actually fitted;
   a separated fit returns `converged = false` with its separated terms
   named, per the ecosystem's separation policy.

# Warnings and their wording

Each step that finds something warns, unless `warn=false` (a bootstrap
replicate's boundary is not a fact about the data). Every warning starts
with `context`, the fitter's name. `estimate` names the estimator
(`"CMPLE"`, `"swap-MPLE"`), and `noun` the rows (`"swap comparisons"`).

Each warning closes with R ergm's own sentence for the same design. `note`
replaces it for a package whose R counterpart behaves differently. It is a
`NamedTuple` with any of these keys:

- `boundary`: the drop (R ergm: "R ergm reports the same");
- `not_varying`: a statistic whose change statistics are all 0;
- `linear_dependence`: a statistic in the span of the ones before it;
- `separation`: the separation warning.

A key that is not given keeps R ergm's sentence; an unknown key is an
`ArgumentError`.

# Result

A `NamedTuple` with:

- `coefficients`, `std_errors`, `var_cov` over all columns, with ∓Inf and
  SE 0 for the dropped ones and `NaN` for the aliased ones;
- `loglik` (the maximised pseudo-log-likelihood), `converged`;
- `boundary`, the dropped `(column, :min | :max)`, and `aliased`, the `NaN`
  columns;
- `fitted`, the columns Newton estimated, and `kept_rows`, the rows they were
  fitted on;
- `verdict`, NetworkCore's `SeparationVerdict` on `X[kept_rows, fitted]`
  (its `terms` index `fitted`), or `nothing` when no column was left to fit;
- `separated` and `separated_terms` (the names of the separated columns),
  read from `verdict`;
- `n_dyads`, the sum of `n_tot`, and `n_kept`, the dyads the finite
  coefficients were estimated on (R's `logLik` nobs, the BIC sample size).

A caller that issues its own warnings passes `warn=false` and reads
`boundary`, `aliased` and `verdict` from the result; it never needs to run
the separation verdict a second time.

`offset` is a fixed per-row addition to the linear predictor (the
contribution of terms whose coefficients are not estimated).

Part of [`ERGM.Extension`](@ref): ERGM's own MPLE and MCMLE start, TERGM's
CMPLE, ERGMMulti's MPLE and ERGMRank's swap MPLE are all this one fit.

# Example
```julia
using ERGM
X = [1.0 0.0; 1.0 1.0]          # two dyad classes: edges only / edges + x
r = ERGM.Extension.mple_fit_design(X, [100.0, 50.0], [20.0, 25.0], ["edges", "x"]; context="my_fit")
r.converged                     # true
r.coefficients[1] ≈ log(20 / 80)            # true: a saturated design, closed-form logits
r.coefficients[2] ≈ log(25 / 25) - log(20 / 80)   # true
r.n_kept == 150                 # true: nothing dropped
r.verdict.separated             # false
# A statistic at the bottom of its range with all-zero change statistics
# (`extreme`), and one that never varies (aliased: NaN)
Z = [1.0 0.0 0.0; 1.0 0.0 0.0]
z = ERGM.Extension.mple_fit_design(Z, [100.0, 50.0], [20.0, 25.0], ["edges", "triangle", "x"];
                                   extreme=[(2, :min)], warn=false)
z.coefficients[2] == -Inf && isnan(z.coefficients[3])   # true
z.coefficients[1] ≈ log(45 / 105)                       # true
z.boundary                      # [(2, :min)]
z.aliased                       # [3]
# A package whose R counterpart does not drop words the warning itself
ERGM.Extension.mple_fit_design([1.0 0.0; 1.0 1.0], [3.0, 2.0], [1.0, 0.0], ["edges", "x"];
    context="my_fit", note=(boundary="my R package returns a finite value instead",))
```
"""
function mple_fit_design(X::Matrix{Float64}, n_tot::Vector{Float64},
                         n_one::Vector{Float64}, names::Vector{String};
                         verbose::Bool=false, maxiter::Int=100, tol::Float64=1e-8,
                         warn::Bool=true, context::AbstractString="mple",
                         estimate::AbstractString="MPLE",
                         noun::AbstractString="dyads",
                         note::NamedTuple=(;),
                         offset::Union{Nothing,Vector{Float64}}=nothing,
                         extreme::AbstractVector{Tuple{Int,Symbol}}=Tuple{Int,Symbol}[])
    p = size(X, 2)
    n_dyads = sum(n_tot)
    notes = _mple_notes(note)

    # 1. Statistics at a boundary of their attainable range: fixed at ∓Inf,
    #    the rest fitted on the rows they do not touch (R's drop=TRUE)
    boundary = boundary_columns(X, n_tot, n_one; fixed=extreme)
    warn && !isempty(boundary) &&
        warn_boundary(names, boundary; context=context, noun=noun, note=notes.boundary)
    dropped = Dict(boundary)
    keep_rows = isempty(dropped) ? collect(axes(X, 1)) :
        [r for r in axes(X, 1) if all(X[r, j] == 0 for j in keys(dropped))]
    cand = [j for j in 1:p if !haskey(dropped, j)]

    # 2. Statistics that do not vary on the rows fitted, or are linear
    #    combinations of the preceding ones: not identifiable. Newton would
    #    stop at its start on the singular information (every coefficient 0,
    #    the identifiable ones included); like R's glm they are reported as
    #    NaN (R: NA) and the rest is fitted without them
    aliased = _aliased_columns(X, keep_rows, cand)
    warn && !isempty(aliased) &&
        _warn_aliased(names, aliased, X, keep_rows; context=context, estimate=estimate,
                      noun=noun, not_varying=notes.not_varying,
                      linear_dependence=notes.linear_dependence)
    keep_cols = setdiff(cand, aliased)

    θ = zeros(p); se = zeros(p); V = zeros(p, p)
    loglik = 0.0
    converged = true
    separated = false
    separated_terms = String[]
    verdict = nothing
    # R's `logLik` nobs attribute: the dyads the dropped columns do not touch
    # (300 of 435 on the docs' separated network) — the BIC sample size
    n_kept = isempty(dropped) ? n_dyads : sum(n_tot[keep_rows]; init=0.0)
    if !isempty(keep_cols)
        whole = isempty(dropped) && isempty(aliased)
        Xr = whole ? X : X[keep_rows, keep_cols]
        tr = whole ? n_tot : n_tot[keep_rows]
        or = whole ? n_one : n_one[keep_rows]
        offr = (offset === nothing || whole) ? offset : offset[keep_rows]
        if verbose
            println("Fitting logistic regression ($(size(Xr, 1)) unique rows, " *
                    "$(Int(n_kept)) dyads" *
                    (isempty(dropped) ? "" :
                     "; $(length(dropped)) coefficient(s) fixed at ±Inf") * ")...")
        end
        # The separation verdict on the design actually fitted (the boundary
        # columns already dropped: run on the full design it would flag
        # them). It depends on the data only — a finite offset cannot change
        # it; the coefficients and standard errors are still computed, for
        # diagnosis
        verdict = logistic_separation(Xr, tr, or)
        f = logistic_derivatives(Xr, tr, or; offset=offr)
        fit = newton_fit(f, _offset_start(Xr, tr, or, offr; maxiter=maxiter, tol=tol);
                         maxiter=maxiter, tol=tol)
        θ[keep_cols] = fit.θ
        se[keep_cols] = fit.se
        V[keep_cols, keep_cols] = fit.vcov
        loglik = fit.loglik
        separated = verdict.separated
        separated_terms = names[keep_cols][verdict.terms]
        separated && warn &&
            warn_separation(context, verdict, names[keep_cols]; estimate=estimate,
                            note=notes.separation)
        converged = fit.converged && !separated
    end
    for (j, side) in dropped
        θ[j] = side === :min ? -Inf : Inf
    end
    for j in aliased
        θ[j] = NaN; se[j] = NaN
        V[j, :] .= NaN; V[:, j] .= NaN
    end
    return (coefficients=θ, var_cov=V, std_errors=se, loglik=loglik,
            converged=converged, separated=separated, separated_terms=separated_terms,
            n_dyads=n_dyads, n_kept=n_kept, boundary=boundary, aliased=aliased,
            fitted=keep_cols, kept_rows=keep_rows, verdict=verdict)
end

# The closing sentences of `mple_fit_design`'s warnings: R ergm's, except
# where the caller's `note` replaces them
const _MPLE_NOTE_KEYS = (:boundary, :not_varying, :linear_dependence, :separation)
function _mple_notes(note::NamedTuple)
    for k in keys(note)
        k in _MPLE_NOTE_KEYS || throw(ArgumentError(
            "mple_fit_design: unknown `note` key :$k; the keys are " *
            join((":$x" for x in _MPLE_NOTE_KEYS), ", ")))
        getfield(note, k) isa AbstractString || throw(ArgumentError(
            "mple_fit_design: `note.$k` must be a string, got $(typeof(getfield(note, k)))"))
    end
    return (boundary=String(get(note, :boundary, _R_BOUNDARY_NOTE)),
            not_varying=String(get(note, :not_varying, _R_NOT_VARYING_NOTE)),
            linear_dependence=String(get(note, :linear_dependence, _R_LINEAR_DEPENDENCE_NOTE)),
            separation=String(get(note, :separation, _R_MPLE_NOT_EXIST)))
end

"""
    _aliased_columns(X, rows, cols) -> Vector{Int}

The columns of `X[rows, cols]` that are not identifiable on those rows: a
column of zeros (the statistic does not vary — every change statistic is 0),
or a column in the span of the columns before it, taken in order — R's
`glm` rule, which reports their coefficients as `NA`. Decided by
Gram–Schmidt with re-orthogonalisation, a residual below `1e-9` of the
column's norm counting as zero.
"""
function _aliased_columns(X::AbstractMatrix, rows::AbstractVector{Int},
                          cols::AbstractVector{Int})
    out = Int[]
    basis = Vector{Float64}[]
    for j in cols
        x = Float64[X[r, j] for r in rows]
        nx = sqrt(sum(abs2, x; init=0.0))
        if nx == 0
            push!(out, j)
            continue
        end
        for _ in 1:2, q in basis
            c = sum(q[i] * x[i] for i in eachindex(x))
            x .-= c .* q
        end
        nr = sqrt(sum(abs2, x; init=0.0))
        if nr <= 1e-9 * nx
            push!(out, j)
        else
            push!(basis, x ./ nr)
        end
    end
    return out
end

# R's two warnings for statistics without a coefficient (`ergm.mple`'s
# "not varying", glm's linear dependence), in the fitter's words; the
# closing parentheses are R ergm's sentences unless the caller replaces them
function _warn_aliased(names::Vector{String}, aliased::Vector{Int}, X::AbstractMatrix,
                       rows::AbstractVector{Int}; context::AbstractString,
                       estimate::AbstractString="MPLE", noun::AbstractString="dyads",
                       not_varying::AbstractString=_R_NOT_VARYING_NOTE,
                       linear_dependence::AbstractString=_R_LINEAR_DEPENDENCE_NOTE)
    zero = [names[j] for j in aliased if all(X[r, j] == 0 for r in rows)]
    comb = [names[j] for j in aliased if !(names[j] in zero)]
    isempty(zero) || @warn "$context: statistic(s) $(join(zero, ", ")) do not vary on " *
        "the $noun fitted (every change statistic is 0), so their coefficients are " *
        "not identifiable: they are reported as NaN and the other coefficients " *
        "are estimated without them ($not_varying). Remove the term(s), or check " *
        "the data: the observed network may sit at an extreme point of the sample space."
    isempty(comb) || @warn "$context: statistic(s) $(join(comb, ", ")) are linear " *
        "combinations of the preceding statistics on the $noun fitted, so the model " *
        "is not identifiable: their coefficients are reported as NaN and the other " *
        "coefficients are the $estimate without them ($linear_dependence). Remove one " *
        "of the dependent terms."
    return nothing
end

# The Newton start of a pseudo-likelihood fit. Without an offset: zero, as
# always. With one, a start reached by CONTINUATION in the offset: from zero
# a large offset saturates the fitted probabilities (η = offset on every row
# it touches), the Hessian vanishes there, and Newton's first step runs off
# to a flat asymptote although the concave objective has a finite maximum
# (a finite `Offset(GWESP(0.0), 1.5)` beside `Triangle()` "converged" at
# triangle = −51 with "The MPLE does not exist"; the maximum is at −4.15).
# So the offset is switched on in eight equal steps, each fit warm-started
# from the last — every step a small perturbation of a solved problem. The
# intermediate fits are silent; the final fit, from this start, reports.
function _offset_start(X, n_tot, n_one, offset; maxiter::Int, tol::Float64)
    θ = zeros(size(X, 2))
    (offset === nothing || all(iszero, offset)) && return θ
    Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        for t in 0:7
            f = logistic_derivatives(X, n_tot, n_one; offset=(t / 8) .* offset)
            fit = newton_fit(f, θ; maxiter=maxiter, tol=tol)
            all(isfinite, fit.θ) && (θ = fit.θ)
        end
    end
    return θ
end

"""
    mple(model::ERGMModel; verbose=false, se=nothing, n_boot=100,
         boot_burnin=nothing, boot_interval=nothing, maxiter=100, tol=1e-8,
         rng=Random.default_rng(), proposal=:tnt, drop=true) -> ERGMResult

Fit an ERGM using Maximum Pseudo-Likelihood Estimation: a logistic regression
of the dyad indicators on the change statistics, solved by Newton–Raphson on
the ecosystem's shared kernel (`NetworkCore.logistic_derivatives` +
`NetworkCore.newton_fit`) to a tolerance of `tol` on the pseudo-log-likelihood.
On the Florentine marriage network the dyad-independent fit agrees with R
`ergm` to 1e-12 on the coefficients and 5e-8 on the standard errors
(provenanced fixture, asserted at 1e-6).

# Arguments
- `model::ERGMModel`: The ERGM model specification
- `verbose::Bool=false`: Print progress information
- `maxiter::Int=100`, `tol::Float64=1e-8`: Newton iteration cap and
  convergence tolerance (see `NetworkCore.newton_fit`). A fit that exhausts
  `maxiter` is returned with `converged = false` and a warning
- `se`: How to compute standard errors. The default (`nothing`) is
  `:hessian`, with the inference it supports decided by the formula — see
  "Inference under dyadic dependence" below:
  - `:hessian` — inverse of the observed pseudo-likelihood information.
    For dyad-independent models the pseudo-likelihood is the true
    likelihood and these are the exact ML standard errors. For models with
    dyad-dependent terms they are the *naive* pseudo-likelihood errors —
    typically anticonservative (too small), because the pseudo-likelihood
    treats dependent dyads as independent observations.
  - `:bootstrap` — parametric bootstrap: simulate `n_boot` networks from
    the model at the MPLE estimate (via the MCMC sampler), refit the MPLE
    on each, and use the empirical covariance of the refitted coefficients.
    Honest under dyad dependence (though the MPLE point estimate itself may
    still be biased; consider `method=:mcmle`). A replicate on which the
    MPLE does not exist — a simulated network with no triangle under
    `Triangle()`, no shared-partner edge under `GWESP`, or a separated
    design — is **excluded** from the covariance (its refit is `-Inf`/NaN,
    which would make every standard error NaN). The standard errors are
    then conditional on a finite refit: the excluded replicates are the
    extreme ones, so the standard errors are biased downward. ONE warning
    names how many of `n_boot` were excluded, `fit.boot_replicates` holds every refit (the
    excluded ones as NaN rows) and `approximations(fit)` records it. At
    least two finite refits are required (`ArgumentError` otherwise). A
    point estimate that itself carries a `±Inf` coefficient (see below)
    cannot be simulated from, so `se=:bootstrap` is refused for it.
- `n_boot::Int=100`: Number of bootstrap replicates (only for `se=:bootstrap`)
- `boot_burnin`, `boot_interval`: MCMC controls for the bootstrap
  simulations; default to the dyad-scaled `mcmle` defaults
  (`20 * n_dyads` and `max(100, n_dyads ÷ 10)`)
- `proposal::Symbol=:tnt`: MH proposal of the bootstrap simulations (see
  [`mh_sample`](@ref))
- `drop::Bool=true`: R's `control.ergm(drop=TRUE)`: a statistic at the
  boundary of its attainable range is fixed at `∓Inf` (below); `drop=false`
  refuses such a model
- `rng::AbstractRNG=Random.default_rng()`: Source of the bootstrap
  simulation randomness (reproducible seeding)

# Returns
- `ERGMResult`: Fitted model results (`se_type` records `:hessian` or
  `:bootstrap`)

# Inference under dyadic dependence

For a formula with a dyad-dependent term (`has_dyad_dependent(model)`), the
naive pseudo-likelihood standard errors are not calibrated: in simulation
(n = 30, `edges + gwesp(0.5)`) 95 % Wald intervals built from them covered
the truth 0.85 and 0.71 of the time, against 0.945 and 0.970 for the
parametric bootstrap. So, **by default, a dyad-dependent MPLE fit reports
its point estimates and naive standard errors but no inference built on
them**: `z_values` and `p_values` are `NaN` (the coefficient table shows
`NaN` in those columns, with a note saying why), `confint` refuses with an
`ArgumentError`, and `approximations(fit)` records the suppression. For
calibrated inference use `se=:bootstrap` (z, p and intervals are then
reported) or `method=:mcmle`. Passing `se=:hessian` **explicitly** is the
written opt-in to R's naive Wald table (z and p from the inverse Hessian),
printed with the pseudo-likelihood caveat. Dyad-independent formulas are
unaffected: there the MPLE is the MLE and its z, p and intervals are exact.

Note: the reported log-likelihood is the maximized *pseudo*-log-likelihood;
AIC/BIC derived from it are only heuristics for dependence models.
`dof(fit)` counts the *finite* coefficients and the BIC sample size is the
number of dyads they were estimated on (R's `logLik` df and nobs
attributes), so after a drop (below) AIC/BIC are R's numbers too.

**Separation** that `_boundary_columns_once` cannot see — a combination of
statistics that predicts every tie — leaves the pseudo-likelihood without a
maximum. R warns "The MPLE does not exist!". `mple` decides it with the
shared `NetworkCore.logistic_separation` verdict, an exact linear programme
on the design (as R's `mple.existence`), and follows the ecosystem's
separation policy: it warns, naming the coefficients that carry the
separating direction; returns the fit with `converged == false` and those
names in `fit.separated_terms`; and withholds inference — z values and
p-values are `NaN` and `confint` returns `NaN` limits — while keeping the
coefficients and standard errors where Newton stopped, for diagnosis. `show`
and `approximations` carry the caveat, and `is_exact(fit)` is `false`.
`se=:bootstrap` is refused on such a fit (there is no estimate to simulate
from).

**A statistic at the boundary of its attainable range** — a `NodeMatch`
with no within-group tie, a `NodeMix` cell of a singleton level, a
`Triangle` on a network with no two-path, … — has no finite MPLE. It is
detected as R does — the observed statistic against the term's attainable
range (`ergm.checkextreme.model`), which also catches a statistic whose
change statistics are all zero — and by the pseudo-likelihood design. As R
ergm does (its default `drop=TRUE`), `mple` warns "observed statistic(s) …
are at their smallest attainable values. Their coefficients will be fixed
at -Inf", returns that coefficient as `-Inf` (`+Inf` at the largest value)
with standard error 0 and p-value 0, and fits the remaining coefficients on
the dyads the dropped statistic does not touch — the exact limit of the
pseudo-likelihood, and R's numbers. `NetworkCore.approximations(fit)` lists
the fixed coefficients; `show` prints a note under the table. `mcmle` does
the same (see there). `drop=false` refuses such a model with an
`ArgumentError` (R's `drop=FALSE` is not implemented).

**A statistic that does not vary** on the dyads fitted (every change
statistic 0, with no attainable bound reached), or that is a linear
combination of the statistics before it, has no identifiable coefficient.
Newton would stop at its start on the singular information and return 0
for every coefficient; instead, as R's `glm` does, the coefficient is
reported as `NaN` (standard error `NaN`; R: `NA`), with R's warning, and
the other coefficients are estimated without it.

Dyads masked as missing (`NetworkCore.set_missing_dyad!`) are excluded from the
pseudo-likelihood: an unobserved tie status is not a response, so those
dyads contribute no logistic-regression row and `nobs` decreases
accordingly. Their face values still condition the change statistics of the
observed dyads. This is the standard *available-case* pseudo-likelihood — a
principled treatment of missing data — so `mple` declares
`NetworkCore.supports_missing(mple) == true` and needs no `missing` keyword;
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
              se::Union{Nothing,Symbol}=nothing,
              n_boot::Int=100,
              boot_burnin::Union{Nothing,Int}=nothing,
              boot_interval::Union{Nothing,Int}=nothing,
              maxiter::Int=100,
              tol::Float64=1e-8,
              rng::AbstractRNG=Random.default_rng(),
              proposal::Symbol=:tnt,
              drop::Bool=true) where {T,D}
    # `se=nothing` (the default) is `:hessian` whose inference is withheld
    # under dyadic dependence; an explicit `se=:hessian` opts in to R's
    # naive Wald table (see "Inference under dyadic dependence")
    naive_opt_in = se === :hessian
    se = something(se, :hessian)
    check_se(se, (:hessian, :bootstrap); context="mple")
    _check_proposal(proposal; context="mple")
    _has_curved(model) && throw(ArgumentError(
        "mple: the formula has a curved term (a decay to be estimated), which the " *
        "pseudo-likelihood cannot fit; use method=:mcmle, or fix the decay " *
        "(`GWESP(decay)`, `GWDegree(decay)`)"))

    fit = _mple_fit(model; verbose=verbose, maxiter=maxiter, tol=tol, drop=drop)
    coefficients = fit.coefficients
    p = length(coefficients)
    offset_mask, _ = _offset_info(model)

    # An unconverged fit is a loud result in this ecosystem, never a silent
    # one: it is still returned (converged=false is recorded), but said. A
    # separated design has already been warned about by `_mple_fit` (the
    # shared separation message); the Newton cap is the other way to get
    # here.
    fit.converged || fit.separated || @warn "mple: the Newton iteration did not " *
        "converge within maxiter=$maxiter (the pseudo-likelihood may be " *
        "unbounded — perfect separation or a degenerate statistic). The " *
        "returned coefficients are the last iterate and `converged == false`; " *
        "check the model for a statistic that is constant or perfectly " *
        "predicts the ties."

    var_cov, std_errors = fit.var_cov, fit.std_errors
    boot_replicates = nothing
    se == :bootstrap && fit.separated && throw(ArgumentError(
        "mple: se=:bootstrap is not available when the MPLE does not exist: the " *
        "pseudo-likelihood is separated (on " *
        join(("`$t`" for t in fit.separated_terms), ", ") * "), so there is no " *
        "estimate to simulate from. Remove, merge or coarsen the separating " *
        "term(s)."))
    if se == :bootstrap
        # A coefficient fixed at ∓Inf cannot be simulated from (the sampler's
        # θ'δ would be NaN wherever the dropped statistic changes), so there
        # is nothing to bootstrap: say so instead of returning NaN errors.
        if any(k -> isinf(coefficients[k]) && !offset_mask[k], 1:p)
            fixed = [model.formula.terms.names[k] for k in 1:p
                     if isinf(coefficients[k]) && !offset_mask[k]]
            throw(ArgumentError(
                "mple: se=:bootstrap is not available when a coefficient is fixed " *
                "at ±Inf by a statistic at the boundary of its attainable range " *
                "($(join(fixed, ", "))): a network cannot be simulated at an " *
                "infinite coefficient. Remove the term (as R's drop=TRUE does) " *
                "or keep the default se=:hessian, which reports standard error 0 " *
                "for the fixed coefficient and the inverse-Hessian errors of the rest."))
        end
        if any(isnan, coefficients)
            bad = [model.formula.terms.names[k] for k in 1:p if isnan(coefficients[k])]
            throw(ArgumentError(
                "mple: se=:bootstrap is not available when a coefficient is not " *
                "identifiable ($(join(bad, ", ")): reported as NaN): remove the " *
                "term(s) and refit."))
        end
        var_cov, std_errors, boot_replicates =
            _mple_bootstrap_cov(model, coefficients; n_boot=n_boot,
                                boot_burnin=boot_burnin, boot_interval=boot_interval,
                                rng=rng, verbose=verbose, proposal=proposal)
    end

    # Z-values and p-values (a coefficient fixed at ∓Inf by a boundary
    # statistic has SE 0: z = ∓Inf and p = 0, as R prints them)
    z_values = coefficients ./ std_errors
    p_values = z_pvalues(z_values)
    for k in eachindex(coefficients)
        isinf(coefficients[k]) && (p_values[k] = 0.0)
    end
    # Under dyadic dependence the naive inverse-Hessian errors under-cover
    # (Wald coverage 0.71–0.85 at 95 % in simulation): unless the caller
    # asked for them in writing, no z, p or interval is built on them
    if se === :hessian && !naive_opt_in && has_dyad_dependent(model)
        for k in eachindex(coefficients)
            (isfinite(coefficients[k]) && !offset_mask[k]) || continue
            z_values[k] = NaN
            p_values[k] = NaN
        end
    end
    # A separated pseudo-likelihood has no finite maximum: no z or p on any
    # coefficient (the ecosystem's separation policy; `confint` returns NaN)
    if fit.separated
        fill!(z_values, NaN)
        fill!(p_values, NaN)
    end

    # AIC and BIC (based on the pseudo-likelihood). The degrees of freedom
    # are the FINITE coefficients and the BIC sample size the dyads they were
    # estimated on — every dyad, or, after R's drop of a boundary statistic,
    # the dyads the dropped term does not touch (`fit.n_kept`): R's
    # `logLik.ergm` df and nobs attributes (df = 1, nobs = 300 of 435 on the
    # docs' separated network; `nobs(fit)` itself stays every observed dyad).
    # (an offset is fixed, not estimated: R's df excludes it)
    k = count(j -> isfinite(coefficients[j]) && !offset_mask[j], 1:p)
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
        boot_replicates,
        nothing,          # no MCMLE stopping rule
        fit.separated_terms
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
# THE sentence every bootstrap caller in the ERGM family uses to disclose what
# excluding failed refits does to the standard errors (in the warning, `show`
# and `approximations`)
const _BOOT_EXCLUSION_BIAS =
    "The standard errors are conditional on a finite refit: the excluded " *
    "replicates are the extreme ones, so the standard errors are biased downward."

# The loop itself is `NetworkCore.bootstrap_cov` — the ONE shared bootstrap of the
# ecosystem (NetworkCore.jl `src/bootstrap.jl`), which the count/rank/multilayer
# MPLEs and REM's repeated control sampling also call. This function supplies
# only the two callbacks that are ERGM's: how to simulate at θ̂, and how to refit.
function _mple_bootstrap_cov(model::ERGMModel, θ̂::Vector{Float64};
                             n_boot::Int, boot_burnin, boot_interval,
                             rng::AbstractRNG, verbose::Bool, proposal::Symbol=:tnt)
    # The same dyad-scaled defaults as every sampler (`mcmc_defaults`)
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
                                       missing=:condition_on_face,
                                       proposal=proposal)

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
    # An `Offset` coordinate is fixed: it is held at 0 in the replicates the
    # covariance is taken over (zero variance, R's zero offset row) and
    # restored to its fixed value in the returned replicate matrix
    mask, vals = _offset_info(model)
    free = findall(!, mask)
    function refit(sim)
        r = _mple_fit(ERGMModel(model.formula, sim; reference=model.reference);
                      warn=false)
        r.converged || return fill(NaN, length(θ̂))
        θb = copy(r.coefficients)
        θb[mask] .= 0.0
        return θb
    end

    boot = bootstrap_cov(refit, simulate, θ̂; n_boot=n_boot, rng=rng)
    replicates = boot.replicates
    ok = [all(isfinite, view(replicates, b, free)) for b in 1:n_boot]
    n_ok = count(ok)
    if any(mask)
        V = zeros(length(θ̂), length(θ̂))
        n_ok >= 2 && (V[free, free] = cov(replicates[ok, free]))
        for b in 1:n_boot
            ok[b] && (replicates[b, mask] .= vals[mask])
        end
        n_ok == n_boot && return V, sqrt.(max.(diag(V), 0.0)), replicates
    else
        n_ok == n_boot && return boot.vcov, boot.se, replicates
    end

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
          "refits. $_BOOT_EXCLUSION_BIAS This is about the simulated " *
          "replicates, not about the observed network. `fit.boot_replicates` " *
          "holds every refit (NaN rows excluded); `approximations(fit)` records " *
          "the exclusion."
    V = zeros(length(θ̂), length(θ̂))
    V[free, free] = cov(replicates[ok, free])
    return V, sqrt.(max.(diag(V), 0.0)), replicates
end

"""
    resolve_method(method::Symbol, model; exact=:mple, mcmc=:mcmle,
                   methods=(exact, mcmc), context="fit_ergm") -> Symbol

The estimator a `method=` keyword selects, with R's default rule for
`method=:auto`: the pseudo-likelihood estimator `exact` when no term of the
model is dyad-dependent (there the pseudo-likelihood IS the likelihood, so
the MPLE is the exact MLE), and the Monte-Carlo MLE `mcmc` otherwise — what
R's `ergm()` does (`estimate = "MLE"`, which for a dyad-independent formula
is computed by the logistic regression). Any other `method` must be one of
`methods` and is returned unchanged; anything else is an `ArgumentError`
naming the choices.

`model` is anything [`has_dyad_dependent`](@ref) has a method for (an
[`ERGMModel`](@ref), a variant's own model type), or a `Bool` that already
says whether the formula is dyad-dependent. The ERGM variants reuse this
rule with their own estimator names, e.g. `exact=:cmple, mcmc=:cmle` for
TERGM's conditional estimators, so `:auto` means the same thing across the
family.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
ERGM.resolve_method(:auto, ERGMModel(ERGMFormula([Edges(), NodeCov(:wealth)]), net))  # :mple
ERGM.resolve_method(:auto, ERGMModel(ERGMFormula([Edges(), Triangle()]), net))        # :mcmle
ERGM.resolve_method(:mple, true)                                     # :mple (explicit choice kept)
ERGM.resolve_method(:auto, true; exact=:cmple, mcmc=:cmle)           # :cmle
```
"""
function resolve_method(method::Symbol, model; exact::Symbol=:mple, mcmc::Symbol=:mcmle,
                        methods=(exact, mcmc), context::AbstractString="fit_ergm")
    method === :auto && return _formula_dependent(model) ? mcmc : exact
    method in methods && return method
    throw(ArgumentError(
        "$context: unknown estimation method $(repr(method)); expected :auto " *
        "(the default: $(repr(exact)) for a dyad-independent formula, where it is " *
        "exact, and $(repr(mcmc)) otherwise, as R does) or one of " *
        join(map(repr, methods), ", ")))
end

_formula_dependent(dependent::Bool) = dependent
_formula_dependent(model) = has_dyad_dependent(model)::Bool

# The keywords each estimator accepts, read from its own method signature so
# the list cannot drift from the code
_estimator_keywords(::Val{:mple}) = Base.kwarg_decl(which(mple, Tuple{ERGMModel}))
_estimator_keywords(::Val{:mcmle}) = Base.kwarg_decl(which(mcmle, Tuple{ERGMModel}))

# A keyword the chosen estimator does not take is refused in words — most
# often an MPLE keyword (`se=:bootstrap`) on a dyad-dependent formula, which
# `method=:auto` sends to the MCMLE
function _check_estimator_keywords(est::Symbol, method::Symbol, keys)
    accepted = _estimator_keywords(Val(est))
    bad = [k for k in keys if !(k in accepted)]
    isempty(bad) && return nothing
    other = est === :mple ? :mcmle : :mple
    other_accepted = _estimator_keywords(Val(other))
    listed = join(("`$k`" for k in bad), ", ")
    why = method === :auto ?
        " method=:auto chose $(repr(est)) because the formula is " *
        (est === :mcmle ? "dyad-dependent (R's ergm() fits the Monte-Carlo MLE there)." :
                          "dyad-independent (the MPLE is the exact MLE there).") : ""
    hint = all(in(other_accepted), bad) ?
        " $(length(bad) == 1 ? "It is a keyword" : "They are keywords") of " *
        "method=$(repr(other)); pass method=$(repr(other)) explicitly to use " *
        "$(length(bad) == 1 ? "it" : "them")." :
        " See `?fit_ergm` for the keywords of each estimator."
    throw(ArgumentError("fit_ergm: keyword $listed is not accepted by " *
                        "method=$(repr(est)).$why$hint"))
end

"""
    fit_ergm(net, terms; method=:auto, kwargs...) -> ERGMResult
    ergm(net, terms; method=:auto, kwargs...) -> ERGMResult

Fit an ERGM to the observed network `net` with the model `terms` — R ergm's
`ergm(net ~ edges + triangle + ...)`. `ergm` is a `const` alias of `fit_ergm`
(`ergm === fit_ergm`): the statnet name and the ecosystem's `fit_<model>`
name are one function.

Builds `ERGMModel(ERGMFormula(terms), net)` (validating every term against
the network — a missing attribute, a directed-only term on an undirected
network, or a two-mode network throws an `ArgumentError` before any fitting)
and dispatches on `method`:

- `method=:auto` (default) — R's rule ([`ERGM.resolve_method`](@ref)): the
  MPLE when no term is dyad-dependent, where the pseudo-likelihood is the
  likelihood and the MPLE is the exact MLE, and the MCMLE otherwise, as R's
  `ergm()` does.
- `method=:mple` — maximum pseudo-likelihood, [`mple`](@ref). Exact for
  dyad-independent formulas; fast but approximate under dyadic dependence
  (see the caveat `show` prints), where z values, p-values and intervals
  are withheld unless `se=:bootstrap` or `se=:hessian` is asked for.
- `method=:mcmle` — Monte-Carlo maximum likelihood, [`mcmle`](@ref).

Every other keyword is forwarded unchanged to the chosen estimator. A
keyword that estimator does not take is an `ArgumentError` saying which
estimator does — e.g. `se=:bootstrap` on a dyad-dependent formula, which
`method=:auto` fits by MCMLE, asks for `method=:mple`:

| keyword | estimator | meaning |
|:--|:--|:--|
| `verbose` | both | progress output |
| `se`, `n_boot`, `boot_burnin`, `boot_interval` | `mple` | `:hessian` or `:bootstrap` standard errors and the bootstrap's controls (replicates without a finite MPLE are excluded, warned once); the default withholds z/p/intervals for a dyad-dependent formula (see [`mple`](@ref)) |
| `maxiter`, `tol` | `mple` | Newton iteration cap / tolerance of the logistic fit |
| `n_samples`, `burnin`, `interval`, `maxiter`, `init` | `mcmle` | MCMC sample size per iteration, dyad-scaled burn-in / thinning, iteration cap, starting coefficients |
| `termination`, `conv_precision`, `conv_confidence`, `max_n_samples` | `mcmle` | the stopping rule (R ergm's `confidence` equivalence test by default) and the sample-size boost cap |
| `conv_threshold`, `hotelling_alpha`, `gamma0`, `max_step_norm` | `mcmle` | the legacy `termination=:hotelling` rule and Hummel step control |
| `drop` | both | R's `drop=TRUE` (default): a statistic at the boundary of its attainable range is fixed at `∓Inf` and the rest estimated, with R's message; `drop=false` refuses such a model |
| `proposal` | both | MH proposal of every simulation: `:spdyad` (R ergm 4's `SPDyad`, the `mcmle` default), `:tnt` (tie/no-tie, the default of the `mple` bootstrap) or `:random` |
| `effective_size` | `mcmle` | ESS-adaptive sampling target (R's `MCMLE.effectiveSize`, 64 by default; `nothing` = fixed `n_samples` draws per iteration) |
| `bridge_rungs`, `bridge_samples` | `mcmle` | path-sampling log-likelihood controls |
| `missing` | `mcmle` | `:error` (default), `:mle` (missing-data maximum likelihood) or `:condition_on_face` for masked dyads |
| `obs_burnin`, `obs_interval` | `mcmle` | controls of the constrained chain under `missing=:mle` |
| `rng` | both | the `AbstractRNG` every Monte-Carlo draw flows from |
| `constraints` | — | statnet's sample-space constraints are not implemented: any non-empty value is an `ArgumentError` |

# Arguments
- `net::Network`: the observed (one-mode) network
- `terms`: a single term, a vector of terms, or a vector mixing terms with
  vectors of terms — `[Edges(), Degree(0:2)]` is spliced into one term per
  degree, as statnet's `degree(0:2)`. Anything else is an `ArgumentError`
  naming the offending element; swapping the two positional arguments
  (`fit_ergm(terms, net)`) is an `ArgumentError` too, not a `MethodError`.
- `method::Symbol=:auto`: `:auto`, `:mple` or `:mcmle`

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
fit = ergm(net, [Edges(), NodeCov(:wealth)])            # dyad-independent: MPLE = MLE
fit.method                                               # :mple
round.(coef(fit); digits=3)                              # [-2.595, 0.011]
fit_dep = fit_ergm(net, [Edges(), GWESP(0.5)]; rng=Xoshiro(1))   # dyad-dependent: MCMLE
fit_dep.method                                           # :mcmle
fit_dep.converged                                        # true
```
"""
function fit_ergm(net::Network, terms::AbstractVector; method::Symbol=:auto,
                  constraints=nothing, kwargs...)
    # statnet's `constraints=` has no counterpart: refuse it in words (the
    # sentence `ERGMFormula` gives), never as a MethodError from the estimator
    (constraints === nothing || (constraints isa AbstractVector && isempty(constraints))) ||
        throw(ArgumentError(_CONSTRAINTS_REFUSAL))
    # an unknown method is refused before the model is built
    method === :auto || resolve_method(method, false)
    formula = ERGMFormula(collect_terms(terms))
    model = ERGMModel(formula, net)
    est = resolve_method(method, model)
    _check_estimator_keywords(est, method, keys(kwargs))
    return est === :mple ? mple(model; kwargs...) : mcmle(model; kwargs...)
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
    ergm(net, terms; method=:auto, kwargs...) -> ERGMResult

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
