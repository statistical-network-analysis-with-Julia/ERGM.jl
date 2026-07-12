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
