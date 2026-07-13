"""
Missing-dyad policy for ERGM routines.

Networks.jl defines the ecosystem contract: `supports_missing` declares that a
routine has a principled treatment of unobserved dyads, and
`require_observed` is the guard every other routine calls so that a masked
dyad is never silently read at face value.

ERGM.jl's two estimators sit on opposite sides of that contract:

- **MPLE** genuinely supports missing data. Masked dyads are unobserved
  responses, so they contribute no row to the pseudo-likelihood design
  matrix (they still *condition* the change statistics of the observed
  dyads). This is the standard available-case pseudo-likelihood, and
  `supports_missing(mple) == true`.

- **MCMLE** and everything built on the MH sampler (`mh_sample`,
  `sample_networks`, `simulate_ergm`, `gof`) do *not*. Holding a masked
  dyad fixed at its stored face value — never toggling it, and counting it
  in the observed sufficient statistics as recorded — targets a
  *different* estimand from the missing-data MLE that statnet computes by
  simulating the unobserved dyads conditionally. That is a legitimate
  approximation, but it must be asked for, not defaulted into. Hence these
  routines take a `missing` keyword:

  - `:error` (default) — refuse a network with masked dyads, via the shared
    `require_observed` error message.
  - `:condition_on_face` — explicit, auditable opt-in to conditioning on
    the face values, with a warning.

Full missing-data maximum likelihood (conditional simulation of the
unobserved dyads) is not implemented; see ERGM.jl issue #4.
"""

"""
    _MISSING_POLICIES

Missing-dyad policies accepted by the MCMC-based routines (`mcmle`,
`mh_sample`, `sample_networks`, `simulate_ergm`, `gof`).

Deliberately *not* `Networks.MISSING_POLICIES`: the opt-in here is spelled
`:condition_on_face` rather than the generic `:face`, because what the
sampler does with a masked dyad (freeze it, and score it, at its stored
value) is a specific ERGM estimand, not merely "read the face value".
"""
const _MISSING_POLICIES = (:error, :condition_on_face)

"""
    _guard_missing(net, policy::Symbol; context) -> Symbol

Enforce `policy` for `net` and return the missing-data method actually in
force: `:none` when the network has no masked dyads, `:condition_on_face`
when it has masked dyads and the caller explicitly opted in.

Throws an `ArgumentError` for an unknown policy, and (via
`Networks.require_observed`, so the message is the shared ecosystem one) for
a masked network under the default `:error` policy.
"""
function _guard_missing(net, policy::Symbol; context::AbstractString)
    policy in _MISSING_POLICIES || throw(ArgumentError(
        "invalid missing-dyad policy $(repr(policy)) for $context; expected " *
        ":error (the default — refuse networks with masked dyads) or " *
        ":condition_on_face (hold masked dyads fixed at their stored face " *
        "value throughout MCMC, and score them at that value)"))

    if policy === :error
        require_observed(net, :error; context=context)
        return :none
    end

    return n_missing_dyads(net) == 0 ? :none : :condition_on_face
end

"""
    _warn_condition_on_face(net, context)

Emit the honest caveat for the `:condition_on_face` opt-in. No-op when the
network has no masked dyads. Called by the user-facing entry points only
(`mcmle`, `simulate_ergm`, bootstrap MPLE), never by the internal sampling
helpers they delegate to, so exactly one warning is emitted per user call.
"""
function _warn_condition_on_face(net, context::AbstractString)
    n = n_missing_dyads(net)
    n == 0 && return nothing
    @warn "The network has $n missing (unobserved) $(n == 1 ? "dyad" : "dyads") " *
          "and `missing=:condition_on_face` was requested, so $context " *
          "conditions on them at their face value: masked dyads are held fixed " *
          "(edge present/absent as stored) during MCMC and enter the observed " *
          "sufficient statistics at that face value. This is a different " *
          "estimand from maximum likelihood under missing data (statnet-style " *
          "missing-data MCMC, which simulates the unobserved dyads " *
          "conditionally); that is not yet implemented. MPLE, by contrast, " *
          "excludes masked dyads from the pseudo-likelihood entirely."
    return nothing
end
