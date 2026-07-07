"""
ERGM network simulation.

Provides functions to simulate networks from fitted ERGM models.
"""

"""
    simulate_ergm(result::ERGMResult; n_sim::Int=1, burnin::Int=10000,
                  interval::Int=1000) -> Vector{Network}

Simulate networks from a fitted ERGM.

# Arguments
- `result::ERGMResult`: Fitted ERGM result
- `n_sim::Int=1`: Number of networks to simulate
- `burnin::Int=10000`: MCMC burn-in steps
- `interval::Int=1000`: Steps between samples

# Returns
- Vector of simulated Network objects
"""
function simulate_ergm(result::ERGMResult{T};
                       n_sim::Int=1,
                       burnin::Int=10000,
                       interval::Int=1000) where T
    model = result.model
    θ = result.coefficients

    return sample_networks(model, θ; n_sim=n_sim, burnin=burnin, interval=interval)
end

"""
    sample_networks(model::ERGMModel, θ::Vector{Float64};
                    n_sim::Int=1, burnin::Int=10000, interval::Int=1000,
                    start_net::Union{Nothing,Network}=nothing) -> Vector{Network}

Sample networks from an ERGM specification.

# Arguments
- `model::ERGMModel`: ERGM model specification
- `θ::Vector{Float64}`: Model coefficients
- `n_sim::Int=1`: Number of networks to sample
- `burnin::Int=10000`: MCMC burn-in steps
- `interval::Int=1000`: Steps between samples
- `start_net::Union{Nothing,Network}=nothing`: Starting network (random if nothing)
"""
function sample_networks(model::ERGMModel{T}, θ::Vector{Float64};
                         n_sim::Int=1,
                         burnin::Int=10000,
                         interval::Int=1000,
                         start_net::Union{Nothing,Network}=nothing) where T
    n = Int(nv(model.network))

    # Initialize network
    if isnothing(start_net)
        current_net = _random_network(T, n; directed=model.directed,
                                      density=network_density(model.network))
    else
        current_net = _copy_network(start_net)
    end

    # Function barrier (see _mcmc_sample)
    return _simulate_run!(current_net, model.formula.terms, θ, model.directed,
                          n_sim, burnin, interval)
end

function _simulate_run!(current_net::Network{T}, terms::TermSet,
                        θ::Vector{Float64}, directed::Bool,
                        n_sim::Int, burnin::Int, interval::Int) where T
    n = nv(current_net)
    networks = Network{T}[]
    delta = Vector{Float64}(undef, length(terms))
    total_steps = burnin + n_sim * interval

    for step in 1:total_steps
        # Propose random dyad toggle
        i = rand(1:n)
        j = rand(1:n)
        while i == j || (!directed && j < i)
            i = rand(1:n)
            j = rand(1:n)
        end

        # Add-direction change statistics; MH ratio is θ'Δ for an addition
        # and −θ'Δ for a removal
        change_stat_all!(delta, terms, current_net, i, j)
        log_accept = dot(θ, delta)

        if has_edge(current_net, i, j)
            log_accept = -log_accept
        end

        if log(rand()) < log_accept
            if has_edge(current_net, i, j)
                rem_edge!(current_net, i, j)
            else
                add_edge!(current_net, i, j)
            end
        end

        # Save network sample
        if step > burnin && (step - burnin) % interval == 0
            push!(networks, _copy_network(current_net))
        end
    end

    return networks
end

"""
    _random_network(T, n; directed=true, density=0.1) -> Network{T}

Create a random network with approximately the specified density.
"""
function _random_network(::Type{T}, n::Int;
                         directed::Bool=true, density::Float64=0.1) where T<:Integer
    net = Network{T}(; n=n, directed=directed)

    for i in 1:n
        j_range = directed ? (1:n) : ((i+1):n)
        for j in j_range
            i == j && continue
            if rand() < density
                add_edge!(net, i, j)
            end
        end
    end

    return net
end
