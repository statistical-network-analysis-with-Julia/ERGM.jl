# ERGM.jl

Exponential Random Graph Models for Julia.

## Overview

ERGM.jl provides tools for fitting, simulating, and diagnosing Exponential-Family Random Graph Models (ERGMs). ERGMs are statistical models for network structure that express the probability of observing a network as a function of network statistics.

This package is a Julia port of the R `ergm` package from the StatNet collection.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Statistical-network-analysis-with-Julia/ERGM.jl")
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
              mcmc_burnin=10000, mcmc_interval=1000)

# Access results
coef(result)         # Coefficients
stderror(result)     # Standard errors
```

## Simulation

```julia
# Simulate networks from fitted model
sim_nets = simulate_ergm(result; n_sim=100)

# Simulate from parameters directly
sim_nets = sample_networks(net, terms, coef;
                           n_samples=100, burnin=1000)
```

## Goodness-of-Fit

```julia
# GOF diagnostics
gof_result = gof(result; statistics=[:degree, :esp, :distance])

# MCMC diagnostics
mcmc_diagnostics(result)
```

## Change Statistics

For efficient MCMC, each term implements `change_stat()`:

```julia
# Change in statistic when toggling edge (i,j)
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

## References

- Hunter, D. R., & Handcock, M. S. (2006). Inference in curved exponential family models for networks. Journal of Computational and Graphical Statistics, 15(3), 565-583.
- Robins, G., Pattison, P., Kalish, Y., & Lusher, D. (2007). An introduction to exponential random graph (p*) models for social networks. Social Networks, 29(2), 173-191.

## License

MIT License
