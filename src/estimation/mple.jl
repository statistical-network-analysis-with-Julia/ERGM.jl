"""
Maximum Pseudo-Likelihood Estimation (MPLE) for ERGMs.

MPLE treats each potential edge as an independent observation and fits
a logistic regression of the dyad indicators on the add-direction change
statistics, each computed conditional on the rest of the observed network.
This is fast but may be biased for models with strong dependencies.
"""

# Numerically stable log(1 + exp(x))
_log1pexp(x::Float64) = x > 35 ? x : (x < -35 ? exp(x) : log1p(exp(x)))

"""
    _z_pvalues(z) -> Vector{Float64}

Two-sided normal p-values `2·P(|Z| ≥ |z|)` computed via the complementary
CDF: the naive `2(1 − cdf)` form underflows to exactly 0 beyond |z| ≈ 8.3,
whereas `ccdf` stays accurate out to |z| ≈ 38. Beyond that the tail
probability is smaller than any Float64, so finite z-statistics are floored
at `floatmin(Float64)` — a finite estimate never has a p-value of exactly
zero. NaN z-values (e.g. from NaN standard errors) stay NaN.
"""
function _z_pvalues(z::AbstractVector{Float64})
    p = 2 .* ccdf.(Normal(), abs.(z))
    @inbounds for k in eachindex(p)
        if p[k] == 0.0 && isfinite(z[k])
            p[k] = floatmin(Float64)
        end
    end
    return p
end

"""
    _mple_data(net, terms::TermSet, directed::Bool)

Build compressed MPLE data: unique change-statistic rows with the number of
dyads (`n_tot`) and the number of observed edges (`n_one`) sharing each row.
Compressing identical rows keeps memory O(unique rows) instead of O(n²).

Dyads masked as missing (see `Network.set_missing_dyad!`) are excluded: their
tie status is unobserved, so they contribute no response row. Their face
values still enter the *predictors* of the remaining dyads, since change
statistics are computed conditional on the rest of the observed network.
"""
function _mple_data(net, terms::TermSet, directed::Bool)
    p = length(terms)
    rows = Dict{Vector{Float64}, Tuple{Float64, Float64}}()

    n = nv(net)
    for i in 1:n
        j_range = directed ? (1:n) : ((i+1):n)
        for j in j_range
            i == j && continue
            is_missing_dyad(net, i, j) && continue

            x = change_stat_all(terms, net, i, j)
            n_tot, n_one = get(rows, x, (0.0, 0.0))
            rows[x] = (n_tot + 1.0, n_one + (has_edge(net, i, j) ? 1.0 : 0.0))
        end
    end

    n_rows = length(rows)
    X = Matrix{Float64}(undef, n_rows, p)
    n_tot = Vector{Float64}(undef, n_rows)
    n_one = Vector{Float64}(undef, n_rows)

    for (r, (x, (tot, one))) in enumerate(rows)
        X[r, :] = x
        n_tot[r] = tot
        n_one[r] = one
    end

    return X, n_tot, n_one
end

# Core MPLE logistic fit. Returns the pieces mple() and the parametric
# bootstrap need: (coefficients, var_cov, std_errors, loglik, converged,
# n_dyads), with var_cov/std_errors from the inverse observed information.
function _mple_fit(model::ERGMModel; verbose::Bool=false)
    net = model.network
    terms = model.formula.terms
    p = length(terms)

    if verbose
        println("Building design matrix...")
    end

    X, n_tot, n_one = _mple_data(net, terms, model.directed)
    n_dyads = sum(n_tot)

    if verbose
        println("Fitting logistic regression ($(size(X, 1)) unique rows, $(Int(n_dyads)) dyads)...")
    end

    # Weighted logistic regression: each unique row r covers n_tot[r] dyads,
    # n_one[r] of which are edges.
    η = Vector{Float64}(undef, size(X, 1))

    function neg_loglik(β)
        mul!(η, X, β)
        ll = 0.0
        @inbounds for r in eachindex(η)
            ll += n_one[r] * η[r] - n_tot[r] * _log1pexp(η[r])
        end
        return -ll
    end

    function grad!(g, β)
        mul!(η, X, β)
        fill!(g, 0.0)
        @inbounds for r in eachindex(η)
            resid = n_one[r] - n_tot[r] / (1 + exp(-η[r]))
            for c in 1:length(g)
                g[c] -= X[r, c] * resid
            end
        end
    end

    result = optimize(neg_loglik, grad!, zeros(p), LBFGS())

    coefficients = Optim.minimizer(result)

    # Covariance from the observed information: H = X' diag(n_tot p(1-p)) X
    mul!(η, X, coefficients)
    p_pred = 1 ./ (1 .+ exp.(-η))
    W = n_tot .* p_pred .* (1 .- p_pred)
    H = X' * (W .* X)

    var_cov, std_errors = try
        V = Matrix(inv(Symmetric(H)))
        V, sqrt.(diag(V))
    catch
        fill(NaN, p, p), fill(NaN, p)
    end

    # Pseudo-log-likelihood at the estimates
    loglik = -neg_loglik(coefficients)

    return (coefficients=coefficients, var_cov=var_cov, std_errors=std_errors,
            loglik=loglik, converged=Optim.converged(result), n_dyads=n_dyads)
end

"""
    mple(model::ERGMModel; verbose=false, se=:hessian, n_boot=100,
         boot_burnin=nothing, boot_interval=nothing,
         rng=Random.default_rng()) -> ERGMResult

Fit an ERGM using Maximum Pseudo-Likelihood Estimation.

# Arguments
- `model::ERGMModel`: The ERGM model specification
- `verbose::Bool=false`: Print progress information
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
    still be biased; consider `method=:mcmle`).
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

Dyads masked as missing (`Network.set_missing_dyad!`) are excluded from the
pseudo-likelihood: an unobserved tie status is not a response, so those
dyads contribute no logistic-regression row and `nobs` decreases
accordingly. Their face values still condition the change statistics of the
observed dyads.
"""
function mple(model::ERGMModel{T}; verbose::Bool=false,
              se::Symbol=:hessian,
              n_boot::Int=100,
              boot_burnin::Union{Nothing,Int}=nothing,
              boot_interval::Union{Nothing,Int}=nothing,
              rng::AbstractRNG=Random.default_rng()) where T
    se in (:hessian, :bootstrap) ||
        throw(ArgumentError("se must be :hessian or :bootstrap, got :$se"))

    fit = _mple_fit(model; verbose=verbose)
    coefficients = fit.coefficients
    p = length(coefficients)

    var_cov, std_errors = fit.var_cov, fit.std_errors
    if se == :bootstrap
        var_cov, std_errors = _mple_bootstrap_cov(model, coefficients;
                                                  n_boot=n_boot,
                                                  boot_burnin=boot_burnin,
                                                  boot_interval=boot_interval,
                                                  rng=rng, verbose=verbose)
    end

    # Z-values and p-values
    z_values = coefficients ./ std_errors
    p_values = _z_pvalues(z_values)

    # AIC and BIC (based on the pseudo-likelihood)
    aic = -2 * fit.loglik + 2 * p
    bic = -2 * fit.loglik + p * log(fit.n_dyads)

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
        se
    )
end

# Parametric-bootstrap covariance of the MPLE: simulate n_boot networks at
# θ̂, refit the MPLE on each (in parallel — the refits are deterministic
# given the simulated networks), and return the empirical covariance and
# SEs of the refitted coefficients.
function _mple_bootstrap_cov(model::ERGMModel, θ̂::Vector{Float64};
                             n_boot::Int, boot_burnin, boot_interval,
                             rng::AbstractRNG, verbose::Bool)
    n_boot >= 2 || throw(ArgumentError("n_boot must be at least 2"))
    p = length(θ̂)
    n_dyads = _n_dyads(model)
    burnin = something(boot_burnin, 20 * n_dyads)
    interval = something(boot_interval, max(100, n_dyads ÷ 10))

    if verbose
        println("Parametric bootstrap: simulating $n_boot networks at the MPLE...")
    end
    sims = sample_networks(model, θ̂; n_sim=n_boot, burnin=burnin,
                           interval=interval, rng=rng)

    boot_coefs = Matrix{Float64}(undef, n_boot, p)
    Threads.@threads for b in 1:n_boot
        # Same (already materialized) formula, simulated network. The
        # simulated networks carry the observed network's attributes, so
        # re-validation/materialization are no-ops.
        boot_model = ERGMModel(model.formula, sims[b]; reference=model.reference)
        boot_coefs[b, :] = _mple_fit(boot_model).coefficients
    end

    V = cov(boot_coefs)
    return Matrix(V), sqrt.(max.(diag(V), 0.0))
end

"""
    fit_ergm(net, terms; method=:mple, kwargs...) -> ERGMResult

Convenience function to fit an ERGM.

# Arguments
- `net`: Network object
- `terms::Vector{<:AbstractERGMTerm}`: Model terms
- `method::Symbol=:mple`: Estimation method (:mple or :mcmle)

# Example
```julia
result = fit_ergm(net, [Edges(), Triangle(), NodeMatch(:gender)])
```
"""
function fit_ergm(net, terms::Vector{<:AbstractERGMTerm};
                  method::Symbol=:mple, kwargs...)
    formula = ERGMFormula(terms)
    model = ERGMModel(formula, net)

    if method == :mple
        return mple(model; kwargs...)
    elseif method == :mcmle
        return mcmle(model; kwargs...)
    else
        throw(ArgumentError("Unknown method: $method"))
    end
end

# Alias
ergm(net, terms; kwargs...) = fit_ergm(net, terms; kwargs...)
