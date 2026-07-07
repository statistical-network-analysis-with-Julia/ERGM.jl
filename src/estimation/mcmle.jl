"""
Monte Carlo Maximum Likelihood Estimation (MCMLE) for ERGMs.

MCMLE uses MCMC sampling to approximate the likelihood function,
providing more accurate estimates than MPLE for models with strong dependencies.
"""

"""
    mcmle(model::ERGMModel; n_samples::Int=1000, burnin::Int=1000,
          interval::Int=100, max_iter::Int=20, tol::Float64=1e-4,
          verbose::Bool=false) -> ERGMResult

Fit an ERGM using Monte Carlo Maximum Likelihood Estimation.

# Arguments
- `model::ERGMModel`: The ERGM model specification
- `n_samples::Int=1000`: Number of MCMC samples per iteration
- `burnin::Int=1000`: Burn-in period
- `interval::Int=100`: Thinning interval
- `max_iter::Int=20`: Maximum MCMLE iterations
- `tol::Float64=1e-4`: Convergence tolerance
- `verbose::Bool=false`: Print progress

# Returns
- `ERGMResult`: Fitted model results. `mcmc_samples` holds the statistics
  sampled at the final coefficients (suitable for `mcmc_diagnostics`);
  samples from earlier iterations are discarded since they target different
  coefficient values.
"""
function mcmle(model::ERGMModel{T};
               n_samples::Int=1000,
               burnin::Int=1000,
               interval::Int=100,
               max_iter::Int=20,
               tol::Float64=1e-4,
               verbose::Bool=false) where T
    # Start with MPLE estimates
    if verbose
        println("Getting initial estimates via MPLE...")
    end

    mple_result = mple(model)
    θ = copy(mple_result.coefficients)

    net = model.network
    terms = model.formula.terms
    p = length(terms)

    # Observed statistics
    obs_stats = compute_all(terms, net)

    converged = false

    for iter in 1:max_iter
        if verbose
            println("MCMLE iteration $iter...")
        end

        # Sample networks from current model
        samples = _mcmc_sample(model, θ, n_samples, burnin, interval)

        # Compute mean and covariance of sampled statistics
        mean_stats = vec(mean(samples, dims=1))
        cov_stats = cov(samples)

        # Check convergence
        diff = obs_stats .- mean_stats
        if maximum(abs.(diff)) < tol
            converged = true
            if verbose
                println("Converged at iteration $iter")
            end
            break
        end

        # Newton-Raphson update
        # θ_new = θ + Σ^(-1) * (obs - E[stats])
        try
            delta = cov_stats \ diff
            θ .+= delta
        catch e
            @warn "Covariance matrix singular at iteration $iter"
            break
        end
    end

    # Compute final statistics
    final_samples = _mcmc_sample(model, θ, n_samples, burnin, interval)
    cov_stats = cov(final_samples)

    # Covariance of θ̂ from the inverse Fisher information
    var_cov, std_errors = try
        V = Matrix(inv(Symmetric(cov_stats)))
        V, sqrt.(diag(V))
    catch
        fill(NaN, p, p), fill(NaN, p)
    end

    z_values = θ ./ std_errors
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(z_values)))

    n = nv(net)
    n_dyads = model.directed ? n * (n - 1) : n * (n - 1) ÷ 2

    # Approximate log-likelihood, normalized against the null (θ = 0) model
    loglik = _approximate_loglik(θ, obs_stats, final_samples, n_dyads)

    aic = -2 * loglik + 2 * p
    bic = -2 * loglik + p * log(n_dyads)

    return ERGMResult(
        model,
        θ,
        std_errors,
        z_values,
        p_values,
        var_cov,
        loglik,
        aic,
        bic,
        :mcmle,
        converged,
        final_samples
    )
end

"""
    _mcmc_sample(model, θ, n_samples, burnin, interval) -> Matrix{Float64}

Generate MCMC samples of network statistics, starting from the observed
network.
"""
function _mcmc_sample(model::ERGMModel{T}, θ::Vector{Float64},
                      n_samples::Int, burnin::Int, interval::Int) where T
    sample_net = _copy_network(model.network)
    # Function barrier: model.formula.terms is not concretely inferable from
    # the model, so dispatch once here and run the loop fully typed.
    return _mcmc_run!(sample_net, model.formula.terms, θ, model.directed,
                      n_samples, burnin, interval)
end

function _mcmc_run!(sample_net, terms::TermSet, θ::Vector{Float64},
                    directed::Bool, n_samples::Int, burnin::Int, interval::Int)
    p = length(terms)
    n = nv(sample_net)

    samples = Matrix{Float64}(undef, n_samples, p)
    total_steps = burnin + n_samples * interval

    current_stats = compute_all(terms, sample_net)
    delta = Vector{Float64}(undef, p)
    sample_idx = 0

    for step in 1:total_steps
        # Propose a random dyad toggle
        i = rand(1:n)
        j = rand(1:n)
        while i == j || (!directed && j < i)
            i = rand(1:n)
            j = rand(1:n)
        end

        # Add-direction change statistics for the dyad
        change_stat_all!(delta, terms, sample_net, i, j)

        # Metropolis–Hastings log acceptance ratio: θ'Δ for an addition,
        # −θ'Δ for a removal
        log_accept = dot(θ, delta)
        if has_edge(sample_net, i, j)
            log_accept = -log_accept
        end

        if log(rand()) < log_accept
            # Accept: toggle edge and update statistics
            if has_edge(sample_net, i, j)
                rem_edge!(sample_net, i, j)
                current_stats .-= delta
            else
                add_edge!(sample_net, i, j)
                current_stats .+= delta
            end
        end

        # Record sample after burn-in, with thinning
        if step > burnin && (step - burnin) % interval == 0
            sample_idx += 1
            samples[sample_idx, :] = current_stats
        end
    end

    return samples
end

"""
    _copy_network(net) -> Network

Create a copy of a network.
"""
function _copy_network(net::Network{T}) where T
    new_net = Network{T}(; n=Int(nv(net)), directed=is_directed(net))

    for e in edges(net)
        add_edge!(new_net, src(e), dst(e))
    end

    return new_net
end

"""
    _approximate_loglik(θ, obs_stats, samples, n_dyads) -> Float64

Approximate the log-likelihood `θ'g(y_obs) − log Z(θ)` via importance
sampling against the null model θ = 0 (whose normalizer is `2^n_dyads`
under the Bernoulli reference):

    log Z(θ) = n_dyads·log 2 − log E_θ[exp(−θ'g(Y))]

The expectation is estimated from statistics sampled at θ. This estimate is
comparable across models but can have high Monte Carlo variance when θ is
far from 0.
"""
function _approximate_loglik(θ::Vector{Float64},
                             obs_stats::Vector{Float64},
                             samples::Matrix{Float64},
                             n_dyads::Int)
    log_weights = -(samples * θ)  # n_samples × 1

    # Log-mean-exp for numerical stability
    max_lw = maximum(log_weights)
    log_mean = max_lw + log(mean(exp.(log_weights .- max_lw)))

    log_normalizer = n_dyads * log(2.0) - log_mean

    return dot(θ, obs_stats) - log_normalizer
end
