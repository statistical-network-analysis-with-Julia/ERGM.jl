# ERGM.jl

*Exponential Random Graph Models for Julia*

A Julia package for fitting, simulating, and diagnosing Exponential-Family Random Graph Models (ERGMs).

## Overview

Exponential Random Graph Models (ERGMs) are statistical models for network data. They express the probability of observing a particular network as a function of network statistics (e.g., number of edges, triangles, or homophily patterns), enabling researchers to test hypotheses about the structural processes that generated an observed network.

ERGM.jl is a port of the R [ergm](https://github.com/statnet/ergm) package from the [StatNet](https://statnet.org/) collection.

### What is an ERGM?

An ERGM models the probability of a network as:

$$P(Y = y) = \frac{\exp\left(\theta^\top g(y)\right)}{c(\theta)}$$

Where:

- $Y$ is the random network, $y$ is the observed network
- $\theta$ is the parameter vector to be estimated
- $g(y)$ is a vector of sufficient statistics (e.g., edge count, triangle count)
- $c(\theta)$ is the normalizing constant

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Network** | A set of nodes and edges (ties) between them |
| **ERGM Term** | A network statistic included in the model (e.g., edges, triangles) |
| **Change Statistic** | The change in a statistic when a single edge is toggled |
| **MPLE** | Maximum Pseudo-Likelihood Estimation — fast approximate method |
| **MCMLE** | Monte Carlo Maximum Likelihood Estimation — accurate iterative method |

### Applications

ERGMs are widely used in:

- **Social network analysis**: Understanding friendship formation, collaboration, and influence
- **Organizational studies**: Modeling inter-firm alliances and intra-organizational communication
- **Public health**: Studying disease transmission networks and intervention strategies
- **Political science**: Analyzing legislative co-sponsorship and international trade networks
- **Ecology**: Modeling species interaction networks and food webs

## Features

- **Rich term library**: Structural terms (edges, triangles, k-stars, GWESP and GWDSP with statnet's directed OTP/ITP/OSP/ISP shared-partner types, GWDegree/GWIDegree/GWODegree, degree/idegree/odegree count terms), nodal attribute terms (nodefactor and nodemix with statnet's per-level expansion, nodecov, nodematch with per-level differential homophily, nodemismatch, absdiff), and dyadic terms (edgecov)
- **Two estimation methods**: MPLE for fast approximation and MCMLE for accurate maximum likelihood
- **Network simulation**: Simulate networks from fitted models via MCMC
- **Goodness-of-fit diagnostics**: Compare observed vs. simulated degree, ESP, and geodesic distance distributions
- **MCMC diagnostics**: Autocorrelation and effective sample size for MCMLE convergence assessment
- **Graphs.jl integration**: Networks implement `AbstractGraph` for full interoperability

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/Networks.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGM.jl")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/ERGM.jl")
```

## Quick Start

```julia
using Networks, ERGM

# Create a network
net = network(30; directed=false)
for (i, j) in [(1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (4,5),
                (5,6), (6,7), (7,8), (8,9), (9,10)]
    add_edge!(net, i, j)
end

# Set a vertex attribute
set_vertex_attribute!(net, :group, Dict(
    1 => "A", 2 => "A", 3 => "A", 4 => "B", 5 => "B",
    6 => "B", 7 => "C", 8 => "C", 9 => "C", 10 => "C"
))

# Define model terms
terms = [Edges(), Triangle(), NodeMatch(:group)]

# Fit the model via MPLE
result = ergm(net, terms; method=:mple)
println(result)

# Simulate networks from fitted model
sim_nets = simulate_ergm(result; n_sim=100)

# Assess goodness of fit
gof_result = gof(result; stats=[:degree, :esp, :distance])
```

## Choosing Terms

| Use Case | Recommended Terms |
|----------|------------------|
| Baseline density | [`Edges`](@ref) |
| Reciprocity (directed) | [`Mutual`](@ref) |
| Triadic closure | [`Triangle`](@ref), [`GWESP`](@ref) |
| Two-path prevalence | [`GWDSP`](@ref) |
| Degree distribution | [`Kstar`](@ref), [`GWDegree`](@ref), [`GWIDegree`](@ref), [`GWODegree`](@ref) |
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
