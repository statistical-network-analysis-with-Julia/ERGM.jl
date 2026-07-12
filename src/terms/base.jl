"""
Base types and interface for ERGM terms.

Defines the abstract term hierarchy and the interface that all
ERGM terms must implement.
"""

"""
    AbstractERGMTerm

Abstract base type for all ERGM terms.

All terms must implement:
- `compute(term, net) -> Float64`: Compute the term statistic for the network
- `change_stat(term, net, i, j) -> Float64`: The add-direction change statistic
  `g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)`, independent of the dyad's current state
- `name(term) -> String`: Return the term name
"""
abstract type AbstractERGMTerm end

"""
    StructuralTerm <: AbstractERGMTerm

Terms based purely on network structure (edges, triangles, etc.).
"""
abstract type StructuralTerm <: AbstractERGMTerm end

"""
    NodalTerm <: AbstractERGMTerm

Terms based on vertex attributes (nodefactor, nodecov, etc.).
"""
abstract type NodalTerm <: AbstractERGMTerm end

"""
    DyadicTerm <: AbstractERGMTerm

Terms based on dyad-level attributes or combinations of vertex attributes.
"""
abstract type DyadicTerm <: AbstractERGMTerm end

"""
    ConstraintTerm <: AbstractERGMTerm

Terms that represent constraints rather than model terms.
"""
abstract type ConstraintTerm <: AbstractERGMTerm end

# Interface functions

"""
    compute(term::AbstractERGMTerm, net) -> Float64

Compute the term statistic for the given network.
"""
function compute(term::AbstractERGMTerm, net)
    error("compute() not implemented for $(typeof(term))")
end

"""
    change_stat(term::AbstractERGMTerm, net, i::Int, j::Int) -> Float64

Compute the add-direction change statistic for dyad (i,j):
`g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)`, i.e. the statistic with edge (i,j) present minus the
statistic with it absent, holding all other dyads at their current values.

The result must not depend on whether edge (i,j) currently exists — this is
the convention required by both the MPLE design matrix and the
Metropolis–Hastings sampler (which negates it for removal proposals).
"""
function change_stat(term::AbstractERGMTerm, net, i::Int, j::Int)
    error("change_stat() not implemented for $(typeof(term))")
end

"""
    name(term::AbstractERGMTerm) -> String

Return a descriptive name for the term.
"""
function name(term::AbstractERGMTerm)
    return string(typeof(term))
end

"""
    TermSet

A collection of ERGM terms.

Terms are stored as a tuple so that `compute_all`/`change_stat_all` compile
to statically dispatched calls per term instead of dynamic dispatch through
an abstractly-typed vector (which would dominate the MCMC inner loop).
"""
struct TermSet{T<:Tuple}
    terms::T
    names::Vector{String}

    function TermSet(terms::T) where {T<:Tuple}
        all(t -> t isa AbstractERGMTerm, terms) ||
            throw(ArgumentError("all elements must be AbstractERGMTerms"))
        names = [name(t) for t in terms]
        new{T}(terms, names)
    end
end

TermSet(terms::Vector{<:AbstractERGMTerm}) = TermSet(Tuple(terms))

Base.length(ts::TermSet) = length(ts.terms)
Base.iterate(ts::TermSet, state=1) = state > length(ts) ? nothing : (ts.terms[state], state + 1)
Base.getindex(ts::TermSet, i) = ts.terms[i]

"""
    compute_all(ts::TermSet, net) -> Vector{Float64}

Compute all term statistics for the network.
"""
function compute_all(ts::TermSet, net)
    return collect(map(t -> compute(t, net), ts.terms))
end

"""
    change_stat_all(ts::TermSet, net, i::Int, j::Int) -> Vector{Float64}

Compute add-direction change statistics for all terms for dyad (i,j).
"""
function change_stat_all(ts::TermSet, net, i::Int, j::Int)
    return collect(map(t -> change_stat(t, net, i, j), ts.terms))
end

"""
    change_stat_all!(dest, ts::TermSet, net, i::Int, j::Int) -> dest

In-place version of [`change_stat_all`](@ref) for use in sampling loops.
"""
function change_stat_all!(dest::AbstractVector{Float64}, ts::TermSet, net, i::Int, j::Int)
    vals = map(t -> change_stat(t, net, i, j), ts.terms)
    for k in eachindex(vals)
        dest[k] = vals[k]
    end
    return dest
end

"""
    summary_stats(net, terms::Vector{<:AbstractERGMTerm}) -> NamedTuple

Compute summary statistics for a network.
"""
function summary_stats(net, terms::Vector{<:AbstractERGMTerm})
    names = [Symbol(name(t)) for t in terms]
    values = [compute(t, net) for t in terms]
    return NamedTuple{Tuple(names)}(values)
end

# ============================================================================
# Model and Result Types
# ============================================================================

"""
    ERGMFormula

Represents an ERGM model specification.

# Fields
- `terms::TermSet`: Model terms
- `constraints::Vector{ConstraintTerm}`: Model constraints
"""
struct ERGMFormula
    terms::TermSet
    constraints::Vector{ConstraintTerm}

    function ERGMFormula(terms::TermSet;
                         constraints::Vector{<:ConstraintTerm}=ConstraintTerm[])
        new(terms, constraints)
    end
end

ERGMFormula(terms::Vector{<:AbstractERGMTerm};
            constraints::Vector{<:ConstraintTerm}=ConstraintTerm[]) =
    ERGMFormula(TermSet(terms); constraints=constraints)

"""
    ERGMModel

An ERGM model specification with observed network.

Construction validates the formula against the network — every
attribute-based term's vertex attribute must exist on the network, and
intrinsically directed terms (e.g. `Mutual`) are rejected on undirected
networks — throwing an `ArgumentError` otherwise. Attribute-based nodal
terms are then *materialized*: their attribute values are snapshotted into
dense typed vectors (see `src/terms/materialize.jl`) so that change
statistics in the estimation and sampling hot loops avoid the untyped
attribute storage. Materialized terms keep the original term names and
semantics.

# Fields
- `formula::ERGMFormula`: Model formula (with materialized terms)
- `network::Network`: Observed network
- `directed::Bool`: Whether the model is for directed networks
- `reference::Symbol`: Reference measure (:bernoulli by default)
"""
struct ERGMModel{T}
    formula::ERGMFormula
    network::Network{T}
    directed::Bool
    reference::Symbol

    function ERGMModel(formula::ERGMFormula, net::Network{T};
                       reference::Symbol=:bernoulli) where T
        _validate_formula(formula.terms, net)
        mformula = ERGMFormula(_materialize(formula.terms, net);
                               constraints=formula.constraints)
        new{T}(mformula, net, is_directed(net), reference)
    end
end

"""
    ERGMResult

Results from fitting an ERGM.

# Fields
- `model::ERGMModel`: The fitted model
- `coefficients::Vector{Float64}`: Estimated coefficients
- `std_errors::Vector{Float64}`: Standard errors
- `z_values::Vector{Float64}`: Z-statistics
- `p_values::Vector{Float64}`: Two-sided p-values
- `vcov::Matrix{Float64}`: Estimated covariance matrix of the coefficients
- `loglik::Float64`: Log-likelihood (or pseudo-log-likelihood)
- `aic::Float64`: AIC
- `bic::Float64`: BIC
- `method::Symbol`: Estimation method (:mple or :mcmle)
- `converged::Bool`: Convergence status
- `mcmc_samples::Union{Nothing, Matrix{Float64}}`: Statistics sampled at the
  final coefficient values (for MCMLE)
- `se_type::Symbol`: How the standard errors were obtained (`:hessian` for
  inverse observed information, `:bootstrap` for parametric bootstrap,
  `:mcmc` for the inverse Fisher information estimated from MCMC samples)
"""
struct ERGMResult{T}
    model::ERGMModel{T}
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    z_values::Vector{Float64}
    p_values::Vector{Float64}
    vcov::Matrix{Float64}
    loglik::Float64
    aic::Float64
    bic::Float64
    method::Symbol
    converged::Bool
    mcmc_samples::Union{Nothing, Matrix{Float64}}
    se_type::Symbol
end

# Backward-compatible constructor from before `se_type` existed
ERGMResult(model, coefficients, std_errors, z_values, p_values, vcov,
           loglik, aic, bic, method, converged, mcmc_samples) =
    ERGMResult(model, coefficients, std_errors, z_values, p_values, vcov,
               loglik, aic, bic, method, converged, mcmc_samples,
               method === :mcmle ? :mcmc : :hessian)

function Base.show(io::IO, result::ERGMResult)
    println(io, "ERGM Results")
    println(io, "============")
    println(io, "Method: $(result.method)")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "AIC: $(round(result.aic, digits=2)), BIC: $(round(result.bic, digits=2))")
    println(io, "Converged: $(result.converged)")
    println(io)
    println(io, "Coefficients:")

    # Shared ecosystem presentation layer (Network.print_coeftable):
    # Estimate / Std.Error / z value / Pr(>|z|) with significance codes
    print_coeftable(io, result.model.formula.terms.names,
                    result.coefficients, result.std_errors, result.p_values;
                    z_values=result.z_values)

    # Honest-uncertainty caveat: pseudo-likelihood fits of dyad-dependent
    # models have suspect inverse-Hessian standard errors (statnet prints an
    # analogous warning). Dyad-independent formulas need no caveat — there
    # the pseudo-likelihood is the likelihood.
    if result.method == :mple &&
       any(is_dyad_dependent(t) for t in result.model.formula.terms)
        println(io)
        if result.se_type == :bootstrap
            println(io, "Note: this model contains dyad-dependent terms and was fit by maximum")
            println(io, "pseudolikelihood (MPLE). Standard errors are parametric-bootstrap")
            println(io, "estimates; the MPLE point estimates may still be biased. Consider")
            println(io, "refitting with method=:mcmle.")
        else
            println(io, "Warning: this model contains dyad-dependent terms and was fit by")
            println(io, "maximum pseudolikelihood (MPLE). The standard errors are based on the")
            println(io, "naive pseudolikelihood and are suspect (typically anticonservative);")
            println(io, "the p-values should not be trusted. Refit with method=:mcmle, or use")
            println(io, "se=:bootstrap for parametric-bootstrap standard errors.")
        end
    end
end

# StatsAPI interface: methods on the shared statistics generics, so results
# interoperate with StatsBase/GLM-style tooling (`coef(fit)`, `aic(fit)`, ...)

"""
    _n_dyads(model::ERGMModel) -> Int

Number of observed free dyads in the model's network (ordered pairs for
directed networks, unordered pairs otherwise), excluding dyads masked as
missing via `Network.set_missing_dyad!` — their tie status is unobserved,
so they are not observations.
"""
function _n_dyads(model::ERGMModel)
    n = Int(nv(model.network))
    total = model.directed ? n * (n - 1) : n * (n - 1) ÷ 2
    return total - n_missing_dyads(model.network)
end

"""
    coef(result::ERGMResult) -> Vector{Float64}

Estimated coefficients of the fitted model (a method of `StatsAPI.coef`).
"""
StatsAPI.coef(result::ERGMResult) = result.coefficients

"""
    stderror(result::ERGMResult) -> Vector{Float64}

Standard errors of the coefficient estimates (a method of
`StatsAPI.stderror`); their type is recorded in `result.se_type`
(`:hessian`, `:bootstrap`, or `:mcmc`).
"""
StatsAPI.stderror(result::ERGMResult) = result.std_errors

"""
    vcov(result::ERGMResult) -> Matrix{Float64}

Variance-covariance matrix of the coefficient estimates (a method of
`StatsAPI.vcov`).
"""
StatsAPI.vcov(result::ERGMResult) = result.vcov
StatsAPI.loglikelihood(result::ERGMResult) = result.loglik
StatsAPI.aic(result::ERGMResult) = result.aic
StatsAPI.bic(result::ERGMResult) = result.bic
StatsAPI.nobs(result::ERGMResult) = _n_dyads(result.model)
StatsAPI.dof(result::ERGMResult) = length(result.coefficients)
