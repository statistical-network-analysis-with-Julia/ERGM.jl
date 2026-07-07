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
    _mple_data(net, terms::TermSet, directed::Bool)

Build compressed MPLE data: unique change-statistic rows with the number of
dyads (`n_tot`) and the number of observed edges (`n_one`) sharing each row.
Compressing identical rows keeps memory O(unique rows) instead of O(n²).
"""
function _mple_data(net, terms::TermSet, directed::Bool)
    p = length(terms)
    rows = Dict{Vector{Float64}, Tuple{Float64, Float64}}()

    n = nv(net)
    for i in 1:n
        j_range = directed ? (1:n) : ((i+1):n)
        for j in j_range
            i == j && continue

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

"""
    mple(model::ERGMModel; verbose::Bool=false) -> ERGMResult

Fit an ERGM using Maximum Pseudo-Likelihood Estimation.

# Arguments
- `model::ERGMModel`: The ERGM model specification
- `verbose::Bool=false`: Print progress information

# Returns
- `ERGMResult`: Fitted model results

Note: the reported log-likelihood is the maximized *pseudo*-log-likelihood;
AIC/BIC derived from it are only heuristics for dependence models.
"""
function mple(model::ERGMModel{T}; verbose::Bool=false) where T
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

    # Z-values and p-values
    z_values = coefficients ./ std_errors
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(z_values)))

    # Pseudo-log-likelihood at the estimates
    loglik = -neg_loglik(coefficients)

    # AIC and BIC (based on the pseudo-likelihood)
    aic = -2 * loglik + 2 * p
    bic = -2 * loglik + p * log(n_dyads)

    return ERGMResult(
        model,
        coefficients,
        std_errors,
        z_values,
        p_values,
        var_cov,
        loglik,
        aic,
        bic,
        :mple,
        Optim.converged(result),
        nothing
    )
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
