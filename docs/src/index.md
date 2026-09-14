# ERGM.jl

Fit probability models for a binary network observed on a fixed set of actors. ERGM.jl combines structural statistics and actor or dyad covariates, estimates their coefficients, and simulates networks for model checking.

| First analysis | Learn the model or data | Reference and detail |
|:--|:--|:--|
| [Fit and check a first model](getting_started.md) | [Choose an estimator](guide/estimation.md) | [Browse model terms](api/terms.md) |

!!! note "Supported scope"

    Estimation supports loop-free, one-mode binary networks. Curved and bipartite ERGMs are not implemented. MPLE is a conditional approximation for dyad-dependent terms; use MCMC-MLE and inspect its convergence diagnostics when that is the intended estimator. Missing dyads require the estimator-specific policies described in the [estimation guide](guide/estimation.md).

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

## Quick Start

Start with a dyad-independent baseline for the bundled Florentine marriage network:

```julia
using Networks, ERGM, Random

net = load_dataset(:florentine_marriage)
fit = fit_ergm(net, [Edges(), NodeCov(:wealth)]; method=:mple)
display(fit)
draws = simulate_ergm(fit; n_sim=3, rng=Xoshiro(42))
println(network_density.(draws))
```

Here the wealth term associates the sum of two families’ wealth with their conditional log-odds of a marriage tie. Dyad independence makes this MPLE objective a likelihood. Three draws illustrate simulation; they are too few for a goodness-of-fit assessment. Dependence terms such as triangles change both the model and the estimation problem.

## Choosing Terms

| Use Case | Recommended Terms |
|----------|------------------|
| Baseline density | [`Edges`](@ref) |
| Reciprocity (directed) | [`Mutual`](@ref) |
| Triadic closure | [`Triangle`](@ref), [`GWESP`](@ref) |
| Two-path prevalence | [`GWDSP`](@ref) |
| Degree distribution | [`Kstar`](@ref), [`GWDegree`](@ref) (undirected); [`OStar`](@ref), [`IStar`](@ref), [`GWIDegree`](@ref), [`GWODegree`](@ref) (directed) |
| Specific degree counts (isolates, ...) | [`Degree`](@ref), [`IDegree`](@ref), [`ODegree`](@ref) |
| Attribute homophily | [`NodeMatch`](@ref), [`AbsDiff`](@ref) |
| Group mixing structure | [`NodeMix`](@ref) |
| Attribute main effects | [`NodeCov`](@ref), [`NodeFactor`](@ref) |
| Dyadic covariates | [`EdgeCov`](@ref) |

## Documentation

```@contents
Pages = [
    "getting_started.md",
    "guide/terms.md",
    "guide/estimation.md",
    "guide/simulation.md",
    "guide/diagnostics.md",
    "api/types.md",
    "api/terms.md",
    "api/estimation.md",
]
Depth = 2
```

## Theoretical Background

### The ERGM Framework

An ERGM specifies the probability of a network $y$ on $n$ nodes as:

$$P_\theta(Y = y) = \frac{\exp\left(\theta^\top g(y)\right)}{c(\theta)}, \quad c(\theta) = \sum_{y' \in \mathcal{Y}} \exp\left(\theta^\top g(y')\right)$$

The normalizing constant $c(\theta)$ sums over all possible networks, making exact computation intractable for all but very small networks. This motivates the two estimation approaches:

- **MPLE** approximates the likelihood by treating dyads as conditionally independent, reducing the problem to logistic regression
- **MCMLE** uses MCMC sampling to approximate the ratio of normalizing constants, iterating via Newton-Raphson until convergence

### Change Statistics

The change statistic $\delta_g(y)_{ij}$ measures how statistic $g$ changes when edge $(i,j)$ is toggled:

$$\delta_g(y)_{ij} = g(y^+_{ij}) - g(y^-_{ij})$$

Change statistics are central to both estimation (MPLE uses them as features) and simulation (MCMC acceptance probabilities depend on them).

## References

1. Hunter, D.R., Handcock, M.S., Butts, C.T., Goodreau, S.M., Morris, M. (2008). ergm: A Package to Fit, Simulate and Diagnose Exponential-Family Models for Networks. *Journal of Statistical Software*, 24(3), 1-29.

2. Robins, G., Pattison, P., Kalish, Y., Lusher, D. (2007). An introduction to exponential random graph (p*) models for social networks. *Social Networks*, 29(2), 173-191.

3. Snijders, T.A.B. (2002). Markov chain Monte Carlo estimation of exponential random graph models. *Journal of Social Structure*, 3(2), 1-40.

4. Strauss, D., Ikeda, M. (1990). Pseudolikelihood estimation for social networks. *Journal of the American Statistical Association*, 85(409), 204-212.

5. Frank, O., Strauss, D. (1986). Markov graphs. *Journal of the American Statistical Association*, 81(395), 832-842.


## Citation

If you use ERGM.jl in your work, please cite it using the entry in
[`CITATION.bib`](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl/blob/main/CITATION.bib):

```biblatex
@misc{SNWJERGMJL,
  author = {{Statistical Network Analysis with Julia}},
  title = {ERGM.jl: Exponential Random Graph Models for Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/ERGM.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/ERGM.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```

## Module

```@docs
ERGM
```
