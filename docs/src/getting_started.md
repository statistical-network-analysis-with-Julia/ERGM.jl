# Getting Started

This tutorial walks through common use cases for ERGM.jl, from basic model fitting to simulation and diagnostics.

## Installation

Install ERGM.jl from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/Network.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGM.jl")
```

## Basic Workflow

The typical ERGM.jl workflow consists of four steps:

1. **Create a network** - Prepare your network data
2. **Define model terms** - Choose which network statistics to include
3. **Fit the model** - Estimate coefficients via MPLE or MCMLE
4. **Assess the fit** - Simulate networks and run goodness-of-fit tests

## Step 1: Create a Network

Networks are represented using the `Network` type from Network.jl:

```julia
using Network, ERGM

# Create an undirected network with 10 nodes
net = network(10; directed=false)

# Add edges
add_edge!(net, 1, 2)
add_edge!(net, 1, 3)
add_edge!(net, 2, 3)
add_edge!(net, 3, 4)
add_edge!(net, 4, 5)

println("Nodes: ", nv(net))   # 10
println("Edges: ", ne(net))   # 5
```

### Setting Vertex Attributes

Vertex attributes are used by nodal terms like `NodeMatch` and `NodeCov`:

```julia
# Categorical attribute
set_vertex_attribute!(net, :gender, Dict(
    1 => "M", 2 => "F", 3 => "M", 4 => "F", 5 => "M",
    6 => "F", 7 => "M", 8 => "F", 9 => "M", 10 => "F"
))

# Continuous attribute
set_vertex_attribute!(net, :age, Dict(
    1 => 25.0, 2 => 30.0, 3 => 28.0, 4 => 35.0, 5 => 22.0,
    6 => 40.0, 7 => 33.0, 8 => 27.0, 9 => 31.0, 10 => 29.0
))
```

### Directed Networks

```julia
# Create a directed network
dnet = network(5; directed=true)
add_edge!(dnet, 1, 2)
add_edge!(dnet, 2, 1)  # Reciprocated tie
add_edge!(dnet, 1, 3)
```

## Step 2: Define Model Terms

Terms capture different structural mechanisms that may shape the observed network:

```julia
# Basic structural model
terms = [
    Edges(),        # Baseline density (like an intercept)
    Triangle(),     # Triadic closure tendency
]
```

### Exploring Available Terms

ERGM.jl provides terms organized by type:

| Category | Terms | Description |
|----------|-------|-------------|
| **Structural** | `Edges`, `Mutual`, `Triangle`, `Kstar`, `TwoPath` | Network topology |
| **Degree counts** | `Degree`, `IDegree`, `ODegree` | Vertices with a given (in-/out-)degree |
| **Geometrically Weighted** | `GWESP`, `GWDSP`, `GWDegree`, `GWIDegree`, `GWODegree` | Downweighted structural terms |
| **Nodal** | `NodeFactor`, `NodeCov`, `NodeMatch`, `NodeMismatch`, `NodeMix`, `AbsDiff` | Vertex attribute effects |
| **Dyadic** | `EdgeCov` | Dyad-level covariate effects |

### Example: Comprehensive Model

```julia
# Structural + attribute model
terms = [
    Edges(),                           # Baseline density
    Triangle(),                        # Triadic closure
    NodeMatch(:gender),                # Gender homophily
    NodeCov(:age),                     # Age effect on tie formation
    AbsDiff(:age),                     # Age similarity effect
]
```

## Step 3: Fit the Model

Use `ergm` (or `fit_ergm`) to estimate the model:

```julia
result = ergm(net, terms; method=:mple)
```

### Estimation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `:mple` | Maximum Pseudo-Likelihood | Fast, good for exploration |
| `:mcmle` | Monte Carlo MLE | Accurate, for final results |

### MPLE (Fast)

```julia
# Quick estimation — treats edges as conditionally independent
result = ergm(net, terms; method=:mple, verbose=true)
```

### MCMLE (Accurate)

```julia
# Iterative MCMC-based estimation
result = ergm(net, terms;
    method = :mcmle,
    n_samples = 1000,
    burnin = 1000,
    interval = 100,
    max_iter = 20,
    tol = 1e-4,
    verbose = true
)
```

### MCMLE Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_samples` | MCMC samples per iteration | 1000 |
| `burnin` | Burn-in steps | 1000 |
| `interval` | Thinning interval | 100 |
| `max_iter` | Maximum Newton-Raphson iterations | 20 |
| `tol` | Convergence tolerance | 1e-4 |

## Step 4: Interpret Results

The result object contains coefficient estimates and test statistics:

```julia
# Print formatted summary table
println(result)

# Output (the shared ecosystem coefficient table):
# ERGM Results
# ============
# Method: mple
# Log-likelihood: -10.797
# AIC: 27.59, BIC: 33.01
# Converged: true
#
# Coefficients:
#                   Estimate  Std.Error  z value  Pr(>|z|)
# edges              -2.4172     0.7513  -3.2172    0.0013 **
# triangle            3.4269     1.3124   2.6111    0.0090 **
# nodematch.gender   -2.0193     1.4598  -1.3833    0.1666
# ---
# Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#
# ... followed by a warning: this model contains dyad-dependent terms
# (Triangle), so the naive MPLE standard errors are suspect — refit with
# method=:mcmle or use se=:bootstrap.
```

### Accessing Results Programmatically

```julia
# Coefficient vector
coef(result)

# Standard errors
stderror(result)

# Variance-covariance matrix
vcov(result)
```

### Interpreting Coefficients

Coefficients are **log-odds ratios** for the conditional probability of an edge:

| Coefficient | Interpretation |
|-------------|----------------|
| θ > 0 | Term increases edge probability |
| θ < 0 | Term decreases edge probability |
| θ = 0 | No effect |
| exp(θ) | Odds ratio for one-unit change in statistic |

**Example interpretations:**

- `edges = -2.3` → Low baseline density (exp(-2.3) ≈ 0.10 odds for each potential edge)
- `triangle = 0.9` → Each shared partner increases odds of a tie by 146% (exp(0.9) ≈ 2.46)
- `nodematch.gender = 0.5` → Same-gender ties are 65% more likely (exp(0.5) ≈ 1.65)

## Complete Example

```julia
using Network, ERGM
using Random
using Statistics

Random.seed!(42)

# Create a small social network
net = network(15; directed=false)

# Add edges forming a clustered structure
for (i, j) in [(1,2), (1,3), (2,3), (2,4), (3,4),    # Cluster 1
                (5,6), (5,7), (6,7), (6,8), (7,8),    # Cluster 2
                (9,10), (10,11), (11,12),              # Cluster 3
                (4,5), (8,9)]                           # Bridge ties
    add_edge!(net, i, j)
end

# Set vertex attributes
set_vertex_attribute!(net, :group, Dict(
    i => (i <= 4 ? "A" : i <= 8 ? "B" : "C") for i in 1:15
))

# Define and fit model
terms = [
    Edges(),
    Triangle(),
    NodeMatch(:group),
]

result = ergm(net, terms; method=:mple)
println(result)

# Simulate from fitted model
sim_nets = simulate_ergm(result; n_sim=50)
println("Mean edges in simulations: ",
    mean(ne(s) for s in sim_nets))

# Goodness of fit
gof_result = gof(result; n_sim=50, stats=[:degree, :esp])
deg = only(s for s in gof_result.statistics if s.name == "degree")
println("Degree GOF p-values: ", deg.p_values)
```

## Network Simulation

Simulate networks from a fitted model to assess whether the model reproduces key features of the observed network:

```julia
# Simulate 100 networks
sim_nets = simulate_ergm(result; n_sim=100, burnin=10000, interval=1000)

# Compare observed vs. simulated edge counts
obs_edges = ne(net)
sim_edges = [ne(s) for s in sim_nets]
println("Observed edges: ", obs_edges)
println("Simulated edges (mean ± sd): ",
    round(mean(sim_edges), digits=1), " ± ",
    round(std(sim_edges), digits=1))
```

## Goodness of Fit

Compare observed network properties to the distribution of properties across simulated networks:

```julia
gof_result = gof(result; n_sim=100, stats=[:degree, :esp, :distance])

# Degree distribution GOF (one GOFStatistic panel per statistic)
deg_gof = only(s for s in gof_result.statistics if s.name == "degree")
println("Observed degree distribution: ", deg_gof.observed)
println("Simulated mean: ", round.(vec(mean(deg_gof.simulated; dims=1)), digits=1))
```

## Best Practices

1. **Start simple**: Begin with `Edges()` only, then add terms incrementally
2. **Use MPLE for exploration**: Switch to MCMLE for final results
3. **Check convergence**: Verify `result.converged == true`
4. **Assess fit**: Always run `gof()` to validate the model
5. **Avoid degeneracy**: Use geometrically weighted terms (`GWESP`, `GWDegree`) instead of raw `Triangle` and `Kstar` for larger networks
6. **Set random seeds**: For reproducibility in MCMLE and simulation

## Next Steps

- Learn about all available [Model Terms](guide/terms.md)
- Understand [Model Estimation](guide/estimation.md) in detail
- Explore [Network Simulation](guide/simulation.md) from fitted models
- Run [Goodness-of-Fit](guide/diagnostics.md) diagnostics
