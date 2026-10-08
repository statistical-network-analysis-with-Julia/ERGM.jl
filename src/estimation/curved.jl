"""
Curved exponential-family terms: geometrically weighted terms whose decay is
ESTIMATED (statnet's `gwesp(decay, fixed=FALSE)`, `gwdegree(decay,
fixed=FALSE)`), and the curved MCMLE that fits them (Hunter & Handcock 2006).

A curved term has two parameters, a coefficient θ and a decay α, and the
model is `exp(θ·g_α(y) + …)` with `g_α` the fixed-decay statistic at α. The
score in (θ, α) is `(g_α − E g_α, θ·(D_α − E D_α))`, `D_α = ∂g_α/∂α`, and the
Fisher information is the covariance of `(g_α, θ·D_α)` — so one iteration of
the curved MCMLE is an ordinary MCMLE iteration on the *working* statistics
`(g_α, θ·D_α)` of the fixed-decay model at the current α: the sampler runs
the fixed-decay model (the derivative statistic carries coefficient 0 and is
only recorded), and the Hummel step and the stopping rule are `mcmle`'s.
"""

# ============================================================================
# Term types
# ============================================================================

"""
    CurvedGWESP <: StructuralTerm
    CurvedGWESP(decay=0.5; type=:OTP)
    GWESP(decay; fixed=false)

Geometrically weighted edgewise shared partners with the decay **estimated**
— statnet's `gwesp(decay, fixed=FALSE)`. The term contributes two
coefficients, named as R names them: `"gwesp"` (the coefficient of the
statistic at the fitted decay) and `"gwesp.decay"`; on a directed network
`"gwesp.<type>"` / `"gwesp.<type>.decay"`. `decay` is the starting value
(the MPLE start is the fixed-decay model's at this value). Fitted by
[`mcmle`](@ref) only — a curved term is not a single statistic, so
`compute`/`change_stat`, `mple` and the samplers refuse it; simulate and
assess a curved model through its fitted result (`simulate_ergm(fit)`,
`gof(fit)`), which holds the statistic at the fitted decay.

# Example
```julia
using ERGM
t = GWESP(0.25; fixed=false)
t isa CurvedGWESP                    # true
name(t)                              # "gwesp"
is_dyad_dependent(t)                 # true
```
"""
struct CurvedGWESP <: StructuralTerm
    decay::Float64
    type::Symbol

    function CurvedGWESP(decay::Real=0.5; type::Symbol=:OTP)
        type in (:OTP, :ITP, :OSP, :ISP) ||
            throw(ArgumentError("type must be :OTP, :ITP, :OSP, or :ISP"))
        new(_check_decay(Float64(decay)), type)
    end
end

"""
    CurvedGWDegree <: StructuralTerm
    CurvedGWDegree(decay=0.5)
    GWDegree(decay; fixed=false)

Geometrically weighted degree with the decay **estimated** — statnet's
`gwdegree(decay, fixed=FALSE)` (undirected networks). Two coefficients,
`"gwdegree"` and `"gwdegree.decay"`, as in R; fitted by [`mcmle`](@ref)
only (see [`CurvedGWESP`](@ref)).

# Example
```julia
using ERGM
t = GWDegree(0.5; fixed=false)
t isa CurvedGWDegree                 # true
name(t)                              # "gwdegree"
```
"""
struct CurvedGWDegree <: StructuralTerm
    decay::Float64

    CurvedGWDegree(decay::Real=0.5) = new(_check_decay(Float64(decay)))
end

const _CurvedTerm = Union{CurvedGWESP,CurvedGWDegree}

name(::CurvedGWESP) = "gwesp"
name(t::CurvedGWESP, net) = is_directed(net) ? "gwesp.$(t.type)" : "gwesp"
name(::CurvedGWDegree) = "gwdegree"
requires_undirected(::CurvedGWDegree) = true
_directed_variant_hint(::CurvedGWDegree) =
    "`GWODegree` / `GWIDegree` with a fixed decay (curved directed degree terms " *
    "are not implemented)"

@noinline _curved_not_a_statistic(t) = throw(ArgumentError(
    "term '$(name(t))' is a curved term (its decay is estimated): it is not a single " *
    "statistic and is fitted by `mcmle` / `fit_ergm(...; method=:mcmle)` only. For a " *
    "statistic, a pseudo-likelihood fit or a simulation use the fixed-decay term " *
    "(`GWESP(decay)`, `GWDegree(decay)`); simulate a fitted curved model with " *
    "`simulate_ergm(fit)`."))
compute(t::_CurvedTerm, net) = _curved_not_a_statistic(t)
change_stat(t::_CurvedTerm, net, i::Int, j::Int) = _curved_not_a_statistic(t)

_has_curved(model::ERGMModel) = any(t -> t isa _CurvedTerm, model.formula.terms.terms)

# The fixed-decay statistic of a curved term at decay α, and its α-derivative
_fixed_at(t::CurvedGWESP, α::Float64) = GWESP(α; type=t.type)
_fixed_at(::CurvedGWDegree, α::Float64) = GWDegree(α)
_deriv_at(t::CurvedGWESP, α::Float64) = _GWESPDecayScore(α, t.type)
_deriv_at(::CurvedGWDegree, α::Float64) = _GWDegreeDecayScore(α)

# ∂/∂α of the GWESP statistic at decay α: the shared-partner code with the
# derivative weights (`_GWDerivKernel`)
struct _GWESPDecayScore <: StructuralTerm
    decay::Float64
    type::Symbol
end
name(t::_GWESPDecayScore) = "gwesp.decay.score.$(_decay_label(t.decay))"
compute(t::_GWESPDecayScore, net) = _gwesp_compute(_GWDerivKernel(t.decay), t.type, net)
change_stat(t::_GWESPDecayScore, net, i::Int, j::Int) =
    _gwesp_change(_GWDerivKernel(t.decay), t.type, net, i, j)

# ∂/∂α of the GWDegree statistic at decay α
struct _GWDegreeDecayScore <: StructuralTerm
    decay::Float64
end
name(t::_GWDegreeDecayScore) = "gwdegree.decay.score.$(_decay_label(t.decay))"
function compute(t::_GWDegreeDecayScore, net)
    K = _GWDerivKernel(t.decay)
    return sum(_spw(K, length(neighbors(net, v))) for v in vertices(net); init=0.0)
end
function change_stat(t::_GWDegreeDecayScore, net, i::Int, j::Int)
    K = _GWDerivKernel(t.decay)
    has_ij = has_edge(net, i, j)
    return _spinc(K, length(neighbors(net, i)) - has_ij) +
           _spinc(K, length(neighbors(net, j)) - has_ij)
end

# What a fitted curved term becomes in the result's model: the fixed-decay
# statistic at the fitted decay under R's label, followed by a placeholder
# for the decay coefficient whose statistic is identically 0 — so that
# `θ'g(y)` of the result's model IS the fitted model, and `simulate_ergm`,
# `gof` and the bridge need no special case.
struct _FittedCurved{T<:AbstractERGMTerm} <: StructuralTerm
    term::T
    label::String
end
name(t::_FittedCurved) = t.label
compute(t::_FittedCurved, net) = compute(t.term, net)
change_stat(t::_FittedCurved, net, i::Int, j::Int) = change_stat(t.term, net, i, j)
requires_undirected(t::_FittedCurved) = requires_undirected(t.term)

struct _DecayParameter <: StructuralTerm
    label::String
end
name(t::_DecayParameter) = t.label
compute(::_DecayParameter, net) = 0.0
change_stat(::_DecayParameter, net, i::Int, j::Int) = 0.0

# ============================================================================
# The curved MCMLE
# ============================================================================

function _mcmle_curved(model::ERGMModel{T,D};
                       n_samples::Int, burnin, interval, maxiter::Int, n_chains::Int,
                       termination::Symbol, conv_precision::Float64,
                       conv_confidence::Float64, n_max::Int,
                       conv_threshold::Float64, hotelling_alpha::Float64,
                       gamma0::Float64, max_step_norm::Float64, init,
                       rng::AbstractRNG, bridge_rungs::Int, bridge_samples,
                       missing::Symbol, proposal::Symbol, effective_size, drop::Bool=true,
                       verbose::Bool) where {T,D}
    net = model.network
    spec = collect(AbstractERGMTerm, model.formula.terms.terms)
    termination === :confidence || throw(ArgumentError(
        "mcmle: curved terms are fitted with termination=:confidence only"))
    (n_missing_dyads(net) == 0 && missing === :error) || throw(ArgumentError(
        "mcmle: curved terms on a network with masked dyads are not implemented; " *
        "fix the decay (`GWESP(decay)`) to use missing=:mle"))
    any(t -> t isa Offset, spec) && throw(ArgumentError(
        "mcmle: offsets together with curved terms are not implemented; fix the " *
        "decay or drop the offset"))

    # Layout of the coefficient vector φ: one entry per ordinary term, two
    # (coefficient, decay) per curved term, in formula order
    names = String[]
    coef_of = Int[]                 # position in φ of term k's coefficient
    decay_of = Int[]                # position of its decay (0 if not curved)
    for t in spec
        push!(names, name(t, net)); push!(coef_of, length(names))
        if t isa _CurvedTerm
            push!(names, name(t, net) * ".decay"); push!(decay_of, length(names))
        else
            push!(decay_of, 0)
        end
    end
    P = length(names)
    decays = [d for d in decay_of if d != 0]

    # The fixed-decay model at given decays, and the working model whose
    # statistics are (…, g_α, D_α, …)
    fixed_model(φ) = ERGMModel(ERGMFormula(AbstractERGMTerm[
        t isa _CurvedTerm ? _fixed_at(t, φ[decay_of[k]]) : t
        for (k, t) in enumerate(spec)]), net)
    function working(φ)
        W = AbstractERGMTerm[]
        θw = Float64[]
        scale = Float64[]
        for (k, t) in enumerate(spec)
            if t isa _CurvedTerm
                α = φ[decay_of[k]]
                push!(W, _fixed_at(t, α)); push!(θw, φ[coef_of[k]]); push!(scale, 1.0)
                push!(W, _deriv_at(t, α)); push!(θw, 0.0); push!(scale, φ[coef_of[k]])
            else
                push!(W, t); push!(θw, φ[coef_of[k]]); push!(scale, 1.0)
            end
        end
        return ERGMModel(ERGMFormula(W), net), θw, scale
    end

    # Start: the fixed-decay MPLE at the terms' starting decays
    φ = zeros(P)
    for (k, t) in enumerate(spec)
        t isa _CurvedTerm && (φ[decay_of[k]] = t.decay)
    end
    fm0 = fixed_model(φ)
    burnin, interval = _resolve_mcmc_controls(fm0, burnin, interval)
    design = _mple_data(net, fm0.formula.terms, D)
    _refuse_no_drop(fm0, design, extreme_statistics(fm0.formula.terms, net);
                    context="mcmle",
                    why="the MCMLE of a curved model cannot hold a statistic fixed " *
                        "at ±Inf (R's drop is not combined with curved terms here)",
                    remedy="Remove the term(s), or fix the decay (`GWESP(decay)`, " *
                           "`GWDegree(decay)`): the fixed-decay MCMLE fixes the " *
                           "coefficient at ±Inf and estimates the rest, as R does.")
    if isnothing(init)
        start = mple_fit_design(design..., fm0.formula.terms.names; warn=false)
        start.separated && throw(ArgumentError(
            "mcmle: the fixed-decay MPLE used as the starting point does not exist " *
            "(separation on $(join(("`$t`" for t in start.separated_terms), ", "))); " *
            "supply a starting point with `init=`"))
        any(isnan, start.coefficients) && throw(ArgumentError(
            "mcmle: statistic(s) " *
            join((n for (n, c) in zip(fm0.formula.terms.names, start.coefficients)
                  if isnan(c)), ", ") *
            " do not vary on the observed network, or are linear combinations of the " *
            "statistics before them, so the fixed-decay MPLE that starts the curved " *
            "MCMLE does not exist; remove the term(s), or supply `init=`"))
        φ[coef_of] = start.coefficients
    else
        length(init) == P || throw(ArgumentError(
            "mcmle: init has length $(length(init)) but the curved model has $P " *
            "coefficients ($(join(names, ", ")))"))
        φ = Vector{Float64}(init)
        all(>=(0), φ[decays]) || throw(ArgumentError("mcmle: a starting decay is negative"))
    end

    # `mcmle`'s sampler (ESS-adaptive by default, chains continued between
    # iterations); the working model changes with the decays, so it is passed
    # at every call
    sampler = mcmle_sampler(fm0; effective_size=effective_size, n_samples=n_samples,
                            max_n_samples=n_max, n_free=P, burnin=burnin,
                            interval=interval, n_chains=n_chains, proposal=proposal,
                            rng=rng)
    # Statistics sampled at φ, on the working scale T = (g, θ·D), with the
    # observed working statistics as the target
    function draw(φ, n)
        wm, θw, scale = working(φ)
        S, cl = sampler.draw(θw, n; model=wm)
        return (samples=S .* scale', chain_lengths=cl,
                target=compute_all(wm.formula.terms, net) .* scale)
    end
    # The curved family's step rule, applied to the Hummel step before the
    # stopping rule sees it. Trust region on the decays: the weights are
    # nonlinear in α, so a Newton step that moves a decay by more than 0.5 is
    # outside the range its linearization describes (a weakly identified
    # decay — its coefficient near 0 — asks for such steps); the whole step
    # is scaled back. A decay stays in its domain, α ≥ 0. Convergence is not
    # declared on a scaled or clamped step. A decay past 10 has run off along
    # a flat direction: beyond α ≈ 10 the weights eᵅ(1 − (1 − e⁻ᵅ)ˢ) are s to
    # nine digits for every s that occurs, so it is not identified by these
    # data, and the iteration stops unconverged.
    function adjust(φ, δ, iter)
        big = maximum(abs(δ[d]) for d in decays)
        ok = big <= 0.5
        ok || (δ = δ .* (0.5 / big))
        for d in decays
            if φ[d] + δ[d] < 0
                δ = copy(δ)
                δ[d] = -φ[d]
                ok = false
            end
        end
        stop = any(φ[d] + δ[d] > 10 for d in decays)
        stop && @warn "Curved MCMLE: a decay has diverged past 10 at iteration $iter — " *
                      "it is not identified by these data (its term's coefficient is " *
                      "near 0, or the term is collinear with another). Stopping; the " *
                      "fit is not converged. Fix the decay instead."
        return δ, ok, stop
    end

    # The MCMLE iteration itself is `mcmle_solve`, the one driver: the
    # Hummel step, R's confidence rule and its sample-size boost, on the
    # working statistics, whose mean deviation is the score in (θ, α) and
    # whose covariance is the Fisher information
    sol = mcmle_solve(draw, φ; labels=names,
                       n_samples=sampler.n_samples,
                       maxiter=maxiter, termination=termination,
                       conv_precision=conv_precision, conv_confidence=conv_confidence,
                       conv_threshold=conv_threshold, hotelling_alpha=hotelling_alpha,
                       gamma0=gamma0, max_step_norm=max_step_norm, max_n_samples=n_max,
                       resize=sampler.resize, adjust=adjust, verbose=verbose,
                       context="Curved MCMLE")
    φ = sol.coef
    converged, γ, tests = sol.converged, sol.step_length, sol.tests
    Tm, cl = sol.final.samples, sol.final.chain_lengths
    convergence = MCMLEConvergence((sol.iterations, γ, tests.t_ratios, tests.hotelling_p,
                                    tests.n_eff))
    termination_report = (rule=termination, p_value=sol.termination_p,
                          precision=conv_precision, confidence=conv_confidence,
                          n_samples=size(Tm, 1))
    vcov_fisher, var_cov, std_errors, mc_se = sol.vcov_fisher, sol.vcov, sol.se, sol.mcmc_se
    z_values = φ ./ std_errors
    p_values = z_pvalues(z_values)

    # The result's model: the fixed-decay statistics at the fitted decays
    # under R's labels, each followed by its decay placeholder
    result_terms = AbstractERGMTerm[]
    for (k, t) in enumerate(spec)
        if t isa _CurvedTerm
            push!(result_terms, _FittedCurved(_fixed_at(t, φ[decay_of[k]]), names[coef_of[k]]))
            push!(result_terms, _DecayParameter(names[decay_of[k]]))
        else
            push!(result_terms, t)
        end
    end
    result_model = ERGMModel(ERGMFormula(result_terms), net)
    result_model.formula.terms.names == names || error("curved result labels out of step")

    converged || @warn "Curved MCMLE did not converge in maxiter=$maxiter iterations " *
        "($(_termination_verdict(termination_report, tests.t_ratios, γ))): the estimates " *
        "are the last iterate and the standard errors are unreliable; increase " *
        "maxiter/n_samples, refit the original model from these coefficients " *
        "(`init=coef(fit)`), or fix the decay"

    if bridge_rungs == 0
        loglik = aic = bic = NaN
    else
        # At the fitted decay the model IS the fixed-decay model: its bridge
        obs = compute_all(result_model.formula.terms, net)
        loglik = _bridge_loglik(result_model, φ, obs; nrungs=bridge_rungs,
                                n_samples=something(bridge_samples, n_samples),
                                burnin=burnin, interval=interval, rng=rng,
                                proposal=proposal)
        aic = -2 * loglik + 2 * P
        bic = -2 * loglik + P * log(n_observed_dyads(model))
    end

    return ERGMResult(result_model, φ, std_errors, z_values, p_values, var_cov, loglik,
                      aic, bic, :mcmle, converged, Tm, :mcmc, :none, vcov_fisher, mc_se,
                      convergence, cl, nothing, termination_report)
end
