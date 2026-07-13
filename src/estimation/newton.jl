"""
Shared Newton–Raphson optimizer with step halving.

Factored out (following `Relevent._newton`) so that packages building
likelihood-based network models on top of ERGM.jl can share one tested
optimizer instead of re-implementing it.
"""

"""
    newton_fit(loglik_grad_hess, θ0::AbstractVector;
               maxiter=100, tol=1e-8, max_halvings=10) -> NamedTuple

Maximize a smooth objective (typically a log-likelihood) by Newton–Raphson
with step halving.

`loglik_grad_hess(θ)` must return a tuple `(ll, grad, hess)`: the objective
value, its gradient vector, and its Hessian matrix at `θ` (the Hessian of
the objective itself, i.e. negative-definite near a maximum). Each Newton
step `−hess \\ grad` is halved up to `max_halvings` times until the
objective does not decrease. Convergence is declared when the objective
change is below `tol` and the gradient norm is below `sqrt(tol)`.

# Arguments
- `loglik_grad_hess`: Function `θ -> (ll, grad, hess)`
- `θ0::AbstractVector`: Starting values (copied, not mutated)

# Keywords
- `maxiter::Int=100`: Maximum Newton iterations
- `tol::Float64=1e-8`: Convergence tolerance on the objective change
- `max_halvings::Int=10`: Maximum step halvings per iteration

# Returns
NamedTuple `(θ, se, vcov, loglik, converged, iterations)`, where
`vcov = pinv(-hess)` at the final iterate (the usual observed-information
covariance when the objective is a log-likelihood) and
`se = sqrt.(abs.(diag(vcov)))`.

# Example
```julia
# Poisson mean via the log-likelihood of k events: ll(θ) = kθ − exp(θ)
k = 7.0
fit = newton_fit(θ -> (k*θ[1] - exp(θ[1]), [k - exp(θ[1])], hcat(-exp(θ[1]))),
                 [0.0])
fit.θ[1] ≈ log(k)   # true
```
"""
function newton_fit(loglik_grad_hess, θ0::AbstractVector{<:Real};
                    maxiter::Int=100, tol::Float64=1e-8, max_halvings::Int=10)
    θ = Vector{Float64}(θ0)
    p = length(θ)
    ll, grad, hess = loglik_grad_hess(θ)
    converged = false
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        step = try
            -hess \ grad
        catch
            break
        end

        # Step halving: shrink the Newton step until the objective does not
        # decrease
        stepsize = 1.0
        ll_new, grad_new, hess_new = ll, grad, hess
        for _ in 1:max_halvings
            ll_new, grad_new, hess_new = loglik_grad_hess(θ .+ stepsize .* step)
            ll_new >= ll && break
            stepsize /= 2
        end

        θ .+= stepsize .* step
        ll_change = abs(ll_new - ll)
        ll, grad, hess = ll_new, grad_new, hess_new

        if ll_change < tol && norm(grad) < sqrt(tol)
            converged = true
            break
        end
    end

    vcov = try
        Matrix{Float64}(pinv(-hess))
    catch
        fill(NaN, p, p)
    end
    se = sqrt.(abs.(diag(vcov)))

    return (θ=θ, se=se, vcov=vcov, loglik=ll, converged=converged,
            iterations=iterations)
end

"""
    logistic_derivatives(X::Matrix{Float64}, y::AbstractVector{Bool};
                         offset=nothing) -> Function

The `(ll, grad, hess)` closure of a logistic log-likelihood on design matrix
`X` and binary response `y`, ready for [`newton_fit`](@ref):

    ℓ(β) = Σ_r [ y_r η_r − log(1 + e^{η_r}) ],   η = Xβ + offset
    ∇ℓ   = X'(y − p),   ∇²ℓ = −X' diag(p(1−p)) X.

Every ERGM-family pseudo-likelihood over dyad-independent rows is this
likelihood — `ERGMMulti`'s MPLE over the within-layer dyads, `TERGM`'s CMPLE
over the free dyads of the auxiliary networks, `ERGMRank`'s swap MPLE over the
(ego, alter-pair) comparisons (with `y ≡ true`: the observed order is always the
"success") — so it lives here, next to `newton_fit`, rather than pasted into
three packages. `offset` (a fixed per-row addition to the linear predictor) is
`ergm`'s offset mechanism: `ERGMMulti` uses it for terms whose coefficients are
held fixed.

**Allocation** (review finding 15): the workspaces are allocated ONCE, when the
closure is built. Each evaluation allocates only the length-`p` gradient and
`p×p` Hessian it returns — never the `n×p` weighted design or a per-row `x * x'`
outer product. Downstream packages pin this with `@allocated` regression tests.
"""
function logistic_derivatives(X::AbstractMatrix{Float64},
                              y::AbstractVector{Bool};
                              offset::Union{Nothing,AbstractVector{Float64}}=nothing)
    n, p = size(X)
    length(y) == n ||
        throw(ArgumentError("y has $(length(y)) entries but X has $n rows"))
    offset === nothing || length(offset) == n ||
        throw(ArgumentError("offset has $(length(offset)) entries but X has $n rows"))

    η = Vector{Float64}(undef, n)
    resid = Vector{Float64}(undef, n)
    WX = Matrix{Float64}(undef, n, p)

    return function (β)
        n == 0 && return (0.0, zeros(p), zeros(p, p))
        mul!(η, X, β)
        ll = 0.0
        @inbounds for r in 1:n
            ηr = offset === nothing ? η[r] : η[r] + offset[r]
            pr = 1.0 / (1.0 + exp(-ηr))
            # log p and log(1−p), each computed on its stable branch
            ll += y[r] ? (ηr < 0 ? ηr - log1p(exp(ηr)) : -log1p(exp(-ηr))) :
                         (ηr < 0 ? -log1p(exp(ηr)) : -ηr - log1p(exp(-ηr)))
            resid[r] = (y[r] ? 1.0 : 0.0) - pr
            w = pr * (1 - pr)
            for k in 1:p
                WX[r, k] = w * X[r, k]
            end
        end
        grad = Vector{Float64}(undef, p)
        mul!(grad, transpose(X), resid)
        hess = Matrix{Float64}(undef, p, p)
        mul!(hess, transpose(X), WX, -1.0, 0.0)
        return ll, grad, hess
    end
end
