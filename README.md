# ERGM.jl


[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.12+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="ERGM.jl icon" width="160">
</p>

Exponential Random Graph Models for Julia.

## Overview

ERGM.jl provides tools for fitting, simulating, and diagnosing Exponential-Family Random Graph Models (ERGMs). ERGMs are statistical models for network structure that express the probability of observing a network as a function of network statistics.

This package is a Julia port of the R `ergm` package from the StatNet collection.

## Installation

Requires Julia 1.12+. ERGM.jl depends on the unregistered
[Network.jl](https://github.com/statistical-network-analysis-with-Julia/Network.jl)
package, which must be added first:

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/Network.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGM.jl")
```

## Features

- **Model terms**: Structural, nodal, and dyadic covariate terms
- **Estimation**: MPLE (fast) and MCMLE (full likelihood)
- **Simulation**: MCMC network simulation
- **Diagnostics**: Goodness-of-fit testing, MCMC diagnostics

## Quick Start

```julia
using Network
using ERGM

# Create observed network
net = Network{Int}(; n=50, directed=false)
# ... add edges ...

# Define model terms
terms = [
    Edges(),
    Triangle(),
    NodeMatch(:gender)
]

# Fit model using MPLE
result = ergm(net, terms; method=:mple)
println(result)

# Simulate from fitted model
sim_nets = simulate_ergm(result; n_sim=100)
```

## Model Terms

### Structural Terms
```julia
Edges()              # Edge count (density)
Mutual()             # Reciprocated edges (directed)
Triangle()           # Triangle count
Kstar(k)             # k-star count
TwoPath()            # Two-path count
GWESP(decay)         # Geometrically weighted ESP
GWDegree(decay)      # Geometrically weighted degree
```

### Nodal Terms
```julia
NodeFactor(:attr)           # Categorical node attribute
NodeCov(:attr)              # Continuous node attribute
NodeMatch(:attr)            # Homophily on attribute
AbsDiff(:attr)              # Absolute difference effect
```

### Dyadic Terms
```julia
EdgeCov(matrix)      # Edge covariate
```

## Model Fitting

```julia
# Maximum Pseudo-Likelihood (fast, approximate)
result = ergm(net, terms; method=:mple)

# Monte Carlo MLE (slower, exact)
result = ergm(net, terms; method=:mcmle,
              burnin=10000, interval=1000)

# Access results
coef(result)         # Coefficients
stderror(result)     # Standard errors
```

## Simulation

```julia
# Simulate networks from fitted model
sim_nets = simulate_ergm(result; n_sim=100)

# Simulate from parameters directly
model = ERGMModel(ERGMFormula(terms), net)
sim_nets = sample_networks(model, coef(result);
                           n_sim=100, burnin=1000)
```

## Goodness-of-Fit

```julia
# GOF diagnostics
gof_result = gof(result; stats=[:degree, :esp, :distance])

# MCMC diagnostics
mcmc_diagnostics(result)
```

## Change Statistics

For efficient MCMC, each term implements `change_stat()`, the add-direction
change statistic `g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)` — the statistic with edge (i,j) present
minus the statistic with it absent. Its value does not depend on whether the
edge currently exists:

```julia
# Change in statistic from adding edge (i,j), given the rest of the network
delta = change_stat(term, net, i, j)
```

## Custom Terms

See ERGMUserterms.jl for templates and utilities for developing custom terms.

## Mathematical Background

An ERGM has the form:

```
P(Y = y) = exp(θ'g(y)) / c(θ)
```

Where:
- `Y` is the random network
- `y` is an observed network
- `θ` is the parameter vector
- `g(y)` is the vector of sufficient statistics
- `c(θ)` is the normalizing constant

## Documentation

For more detailed documentation, see:

- [Stable Documentation](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/stable/)
- [Development Documentation](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/dev/)

## References

1. Hunter, D. R., & Handcock, M. S. (2006). Inference in curved exponential family models for networks. *Journal of Computational and Graphical Statistics*, 15(3), 565-583.

2. Robins, G., Pattison, P., Kalish, Y., & Lusher, D. (2007). An introduction to exponential random graph (p*) models for social networks. *Social Networks*, 29(2), 173-191.

3. Hunter, D. R., Handcock, M. S., Butts, C. T., Goodreau, S. M., & Morris, M. (2008). ergm: A Package to Fit, Simulate and Diagnose Exponential-Family Models for Networks. *Journal of Statistical Software*, 24(3), 1-29.

## License

MIT License - see [LICENSE](LICENSE) for details.
