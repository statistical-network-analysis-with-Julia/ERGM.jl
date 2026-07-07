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

    function ERGMFormula(terms::Vector{<:AbstractERGMTerm};
                         constraints::Vector{<:ConstraintTerm}=ConstraintTerm[])
        new(TermSet(terms), constraints)
    end
end

"""
    ERGMModel

An ERGM model specification with observed network.

# Fields
- `formula::ERGMFormula`: Model formula
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
        new{T}(formula, net, is_directed(net), reference)
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
end

function Base.show(io::IO, result::ERGMResult)
    println(io, "ERGM Results")
    println(io, "============")
    println(io, "Method: $(result.method)")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "AIC: $(round(result.aic, digits=2)), BIC: $(round(result.bic, digits=2))")
    println(io, "Converged: $(result.converged)")
    println(io)
    println(io, "Coefficients:")
    println(io, "-"^60)

    term_names = result.model.formula.terms.names
    for (i, tname) in enumerate(term_names)
        sig = result.p_values[i] < 0.001 ? "***" :
              result.p_values[i] < 0.01 ? "**" :
              result.p_values[i] < 0.05 ? "*" :
              result.p_values[i] < 0.1 ? "." : ""
        println(io, "$(rpad(tname, 20)) $(lpad(round(result.coefficients[i], digits=4), 10)) " *
                    "$(lpad(round(result.std_errors[i], digits=4), 10)) " *
                    "$(lpad(round(result.p_values[i], digits=4), 10)) $sig")
    end
    println(io, "-"^60)
    println(io, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
end

# Accessor functions
coef(result::ERGMResult) = result.coefficients
stderror(result::ERGMResult) = result.std_errors
vcov(result::ERGMResult) = result.vcov
