"""
Missing-dyad policy for ERGM routines.

Networks.jl defines the ecosystem contract: `supports_missing` declares that a
routine has a principled treatment of unobserved dyads, `require_observed` is
the guard every other routine calls so that a masked dyad is never silently
read at face value, and `missing_policies(f)` is the routine-level vocabulary
— the policies `f`'s `missing=` keyword actually accepts — that tooling such
as the capability matrix prints instead of assuming `:face` everywhere.

ERGM.jl offers three treatments of a masked dyad, and never defaults into
reading its face value:

- **MPLE** genuinely supports missing data. Masked dyads are unobserved
  responses, so they contribute no row to the pseudo-likelihood design
  matrix (they still *condition* the change statistics of the observed
  dyads). This is the standard available-case pseudo-likelihood, and
  `supports_missing(mple) == true`; `mple` has no `missing=` keyword, so
  `missing_policies(mple) == (:error,)` (the default).

- **MCMLE** takes `missing::Symbol` with three policies,
  `missing_policies(mcmle) == (:error, :condition_on_face, :mle)`:

  - `:error` (default) — refuse a network with masked dyads, via the shared
    `require_observed` error message plus bullets naming the ERGM opt-ins.
  - `:mle` — **missing-data maximum likelihood** (Handcock & Gile 2010;
    what statnet's `ergm()` does with NA ties): the masked dyads are
    integrated out of the likelihood `Z_obs(θ)/Z(θ)`. A second, constrained
    MH chain that toggles only the masked dyads estimates
    `E[g(Y) | Y_obs]`, which replaces the observed statistics as the MCMLE
    target; the Fisher information is `Var[g] − Var[g | Y_obs]`. Use it
    whenever the masked dyads are genuinely unobserved and you want the
    same estimand as R. The fit records `missing_method = :mle`.
  - `:condition_on_face` — hold each masked dyad fixed at its stored face
    value (never toggled, scored as recorded). A *different estimand* —
    the model conditional on the face values — which is what you want only
    when the stored values are the truth by construction (a structural
    zero, a tie fixed by design). Explicit, auditable, warned.

- **Everything built on the MH sampler** (`mh_sample`, `sample_networks`,
  `simulate_ergm`, `gof`) can only *freeze* a masked dyad at its stored
  face value: a simulated network has a value at every dyad, so simulation
  "under missing data" is not defined and `missing=:mle` is refused with a
  message saying so. These routines declare
  `missing_policies(f) == (:error, :condition_on_face)`.

  The generic `:face` is deliberately **not** accepted anywhere, and the
  refusal message does not advertise it (`require_observed(...;
  face_ok=false)`): suggesting a keyword value the routine then rejects is
  its own small lie (panel 2026-09, item 5).
"""

"""
    _MISSING_POLICIES

Missing-dyad policies accepted by the sampling routines (`mh_sample`,
`sample_networks`, `simulate_ergm`, `gof`) — the value each of them returns
from `Networks.missing_policies`. `mcmle` accepts these plus `:mle`
([`_MCMLE_MISSING_POLICIES`](@ref)).

Deliberately *not* `Networks.MISSING_POLICIES`: the opt-in here is spelled
`:condition_on_face` rather than the generic `:face`, because what the
sampler does with a masked dyad (freeze it, and score it, at its stored
value) is a specific ERGM estimand, not merely "read the face value".
"""
const _MISSING_POLICIES = (:error, :condition_on_face)

"""
    _MCMLE_MISSING_POLICIES

Missing-dyad policies accepted by `mcmle` — the sampler policies plus
`:mle`, missing-data maximum likelihood (the masked dyads integrated out by
a constrained chain; see the module docs and [`mcmle`](@ref)).
"""
const _MCMLE_MISSING_POLICIES = (:error, :condition_on_face, :mle)

# The bullets appended to the shared `require_observed` refusal: the escape
# hatches ERGM's routines actually offer.
const _CONDITION_ON_FACE_HINT =
    "  • pass `missing=:condition_on_face` to hold the masked dyads fixed at " *
    "their stored face value throughout MCMC and score them at that value " *
    "(a different estimand from missing-data maximum likelihood; warned)."
const _MLE_HINT =
    "  • pass `missing=:mle` for missing-data maximum likelihood: the masked " *
    "dyads are integrated out of the likelihood by a constrained MCMC chain " *
    "(Handcock & Gile 2010; what R ergm does with NA ties).\n"

# The refusal every sampling routine gives `missing=:mle`
_mle_not_defined_message(context) =
    "$context: `missing=:mle` is an estimation policy of `mcmle` only. " *
    "Missing-data maximum likelihood integrates the masked dyads out of the " *
    "likelihood, but simulation under missing data is not defined for " *
    "$context: a simulated network has a value at every dyad, so the " *
    "sampler would have to invent one for each masked dyad. Fit with " *
    "`mcmle(model; missing=:mle)`; to simulate from the fitted coefficients " *
    "either pass `missing=:condition_on_face` (masked dyads frozen at their " *
    "stored face value, warned) or simulate on a copy of the network with " *
    "`clear_missing_dyads!` applied (every dyad free, face values as data)."

"""
    _guard_missing(net, policy::Symbol; context, policies=_MISSING_POLICIES) -> Symbol

Enforce `policy` for `net` and return the missing-data method actually in
force: `:none` when the network has no masked dyads, otherwise the policy
the caller explicitly opted into (`:condition_on_face`, or `:mle` when
`policies` admits it — `mcmle` passes [`_MCMLE_MISSING_POLICIES`](@ref)).

Throws an `ArgumentError` for an unknown policy (naming the ones the routine
accepts), for `:mle` on a routine that does not estimate (the message says
simulation under missing data is not defined for it), and for a masked
network under the default `:error` policy. The latter message is the shared
ecosystem one from `Networks.require_observed` — called with
`face_ok=false`, because ERGM's routines do not take `:face` — plus bullets
naming the opt-ins the routine *does* take.
"""
function _guard_missing(net, policy::Symbol; context::AbstractString,
                        policies::Tuple=_MISSING_POLICIES)
    if policy === :mle && !(:mle in policies)
        throw(ArgumentError(_mle_not_defined_message(context)))
    end
    policy in policies || throw(ArgumentError(
        "invalid missing-dyad policy $(repr(policy)) for $context; expected " *
        ":error (the default — refuse networks with masked dyads), " *
        ":condition_on_face (hold masked dyads fixed at their stored face " *
        "value throughout MCMC, and score them at that value)" *
        (:mle in policies ? " or :mle (missing-data maximum likelihood: the " *
                            "masked dyads are integrated out by a constrained " *
                            "chain)" : "")))

    if policy === :error
        # The ONE shared refusal, with ERGM's own opt-ins appended as bullets
        # through `hint=` (no catch-and-rethrow of the message text)
        hint = :mle in policies ? _MLE_HINT * _CONDITION_ON_FACE_HINT :
                                  _CONDITION_ON_FACE_HINT
        require_observed(net, :error; context=context, face_ok=false, hint=hint)
        return :none
    end

    return n_missing_dyads(net) == 0 ? :none : policy
end

"""
    _warn_condition_on_face(net, context)

Emit the honest caveat for the `:condition_on_face` opt-in. No-op when the
network has no masked dyads. Called by the user-facing entry points only
(`mcmle`, `simulate_ergm`, bootstrap MPLE), never by the internal sampling
helpers they delegate to, so exactly one warning per user call.
"""
function _warn_condition_on_face(net, context::AbstractString)
    n = n_missing_dyads(net)
    n == 0 && return nothing
    @warn "The network has $n missing (unobserved) $(n == 1 ? "dyad" : "dyads") " *
          "and `missing=:condition_on_face` was requested, so $context " *
          "conditions on them at their face value: masked dyads are held fixed " *
          "(edge present/absent as stored) during MCMC and enter the observed " *
          "sufficient statistics at that face value. This is a different " *
          "estimand from maximum likelihood under missing data — for that, " *
          "fit with `mcmle(model; missing=:mle)`, which integrates the masked " *
          "dyads out with a constrained chain (what R ergm does with NA " *
          "ties). MPLE, by contrast, excludes masked dyads from the " *
          "pseudo-likelihood entirely."
    return nothing
end
