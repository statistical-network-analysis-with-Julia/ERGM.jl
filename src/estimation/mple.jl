"""
Maximum Pseudo-Likelihood Estimation (MPLE) for ERGMs.

MPLE treats each potential edge as an independent observation and fits
a logistic regression model. This is fast but may be biased for models
with strong dependencies.
"""

using Distributions
using Optim
using LinearAlgebra

"""
    mple(model::ERGMModel; verbose::Bool=false) -> ERGMResult

Fit an ERGM using Maximum Pseudo-Likelihood Estimation.

# Arguments
- `model::ERGMModel`: The ERGM model specification
- `verbose::Bool=false`: Print progress information

# Returns
- `ERGMResult`: Fitted model results
"""
function mple(model::ERGMModel{T}; verbose::Bool=false) where T
    net = model.network
    terms = model.formula.terms
    n = nv(net)
    p = length(terms)

    # Build design matrix and response
    if verbose
        println("Building design matrix...")
    end

    # For directed networks: n*(n-1) potential edges
    # For undirected: n*(n-1)/2 potential edges
    if model.directed
        n_dyads = n * (n - 1)
    else
        n_dyads = n * (n - 1) ÷ 2
    end

    X = zeros(n_dyads, p)
    y = zeros(n_dyads)

    idx = 0
    for i in 1:n
        j_range = model.directed ? (1:n) : ((i+1):n)
        for j in j_range
            i == j && continue

            idx += 1
            y[idx] = has_edge(net, i, j) ? 1.0 : 0.0

            # Compute change statistics
            # Note: we compute change stats as if adding edge to empty network
            # This is an approximation - proper MPLE uses conditional stats
            for (t_idx, term) in enumerate(terms)
                X[idx, t_idx] = change_stat(term, net, i, j)
            end
        end
    end

    if verbose
        println("Fitting logistic regression...")
    end

    # Fit logistic regression via maximum likelihood
    # Negative log-likelihood
    function neg_loglik(β)
        η = X * β
        # Numerical stability
        η = clamp.(η, -500, 500)
        ll = sum(y .* η .- log1p.(exp.(η)))
        return -ll
    end

    # Gradient
    function grad!(g, β)
        η = X * β
        η = clamp.(η, -500, 500)
        p_pred = 1 ./ (1 .+ exp.(-η))
        residuals = y .- p_pred
        g .= -X' * residuals
    end

    # Optimize
    result = optimize(neg_loglik, grad!, zeros(p), LBFGS())

    coefficients = Optim.minimizer(result)

    # Compute standard errors from Hessian
    # H = X' * diag(p*(1-p)) * X
    η = X * coefficients
    η = clamp.(η, -500, 500)
    p_pred = 1 ./ (1 .+ exp.(-η))
    W = p_pred .* (1 .- p_pred)

    # Hessian approximation
    H = X' * (W .* X)

    # Standard errors
    try
        var_cov = inv(H)
        std_errors = sqrt.(diag(var_cov))
    catch
        std_errors = fill(NaN, p)
    end

    # Z-values and p-values
    z_values = coefficients ./ std_errors
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(z_values)))

    # Log-likelihood at estimates
    loglik = -neg_loglik(coefficients)

    # AIC and BIC
    aic = -2 * loglik + 2 * p
    bic = -2 * loglik + p * log(n_dyads)

    return ERGMResult(
        model,
        coefficients,
        std_errors,
        z_values,
        p_values,
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
