# Getting Started

Prepare a binary network, choose terms, fit a model, and check simulated networks against the observation. The examples distinguish a quick pseudo-likelihood fit from MCMC-MLE, whose convergence and simulation diagnostics are part of interpreting the result.

!!! note "Before you begin"

    Estimation supports loop-free, one-mode binary networks. Curved and bipartite ERGMs are not implemented. MPLE is a conditional approximation for dyad-dependent terms; use MCMC-MLE and inspect its convergence diagnostics when that is the intended estimator. Missing dyads require the estimator-specific policies described in the [estimation guide](guide/estimation.md).

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

Run the blocks below in order in that environment. They build on variables from earlier steps; stochastic examples use seeded random number generators where shown.

## Basic Workflow

The typical ERGM.jl workflow consists of four steps:

1. **Create a network** - Prepare your network data
2. **Define model terms** - Choose which network statistics to include
3. **Fit the model** - Estimate coefficients via MPLE or MCMLE
4. **Assess the fit** - Simulate networks and run goodness-of-fit tests

## Step 1: Create a Network

Networks are represented using the `Network` type from Networks.jl:

```julia
using Networks, ERGM
using Random

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
| **Structural** | `Edges`, `Mutual`, `Triangle`, `Kstar` (undirected), `OStar`/`IStar` (directed), `TwoPath` | Network topology |
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
| `:mple` | Maximum pseudo-likelihood | Exact objective for dyad-independent models; conditional approximation otherwise |
| `:mcmle` | Monte Carlo maximum likelihood | Fits the likelihood with MCMC error; requires convergence and mixing checks |

### MPLE

```julia
# Fit the product of dyad-conditional probabilities
result = ergm(net, terms; method=:mple, verbose=true)
```

### MCMLE

```julia
# Iterative MCMC-based estimation. (On this 10-node toy network a model with
# `Triangle()` is degenerate — the sampler collapses and `mcmle` says so
# loudly — so the MCMLE example uses the dyad-independent homophily model;
# see the Complete Example below and the estimation guide for real data.)
result = ergm(net, [Edges(), NodeMatch(:gender)];
    method = :mcmle,
    n_samples = 1000,
    maxiter = 20,
    rng = Xoshiro(42),
    verbose = true
)
```

### MCMLE Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_samples` | MCMC samples per iteration (over all chains) | 1000 |
| `n_chains` | Independent chains per iteration (parallel with threads; same result at any thread count) | 1 |
| `burnin` | Burn-in steps per chain | `20 * n_dyads` |
| `interval` | Thinning interval | `max(100, n_dyads ÷ 10)` |
| `maxiter` | Maximum MCMLE (Newton-Raphson) iterations | 20 |
| `bridge_rungs` | Path-sampling rungs for the log-likelihood; `0` skips it (`loglik`/AIC/BIC `NaN`, everything else identical) | 16 |
| `conv_threshold`, `hotelling_alpha` | Convergence tests (per-statistic t-ratios, Hotelling T²) | 0.1, 0.05 |
| `rng` | The `AbstractRNG` every draw flows from | `Random.default_rng()` |

(There is no `tol`: convergence is a statistical test, not a tolerance. A fit
that exhausts `maxiter` is returned with `converged == false`, a warning
quoting the last max t-ratio and Hotelling p-value, a caveat line in `show`,
and the numbers in `result.mcmc_convergence`; continue it with
`init=coef(result)`. The standard errors of an MCMLE fit include the
Monte-Carlo error of the estimate — `show` prints the share as "MCMC %",
`mcmc_se(result)` returns it.)

## Step 4: Interpret Results

The result object contains coefficient estimates and test statistics:

```julia
# Print formatted summary table
println(result)
```

`result` is the MCMLE fit of `[Edges(), NodeMatch(:gender)]` from the block
above, and this is what it prints (the shared ecosystem coefficient table,
followed by R's "MCMC %" line for an MCMLE fit):

```text
ERGM Results
============
Method: mcmle
Log-likelihood: -14.9621
AIC: 33.92, BIC: 37.54
Converged: true

Coefficients:
                  Estimate  Std.Error  z value  Pr(>|z|)
edges              -1.6582     0.5220  -3.1769    0.0015 **
nodematch.gender   -1.2862     1.1446  -1.1238    0.2611
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

MCMC % of the standard error (100·(se − se_fisher)/se): edges 0, nodematch.gender 0
```

An MPLE fit of a model with a dyad-dependent term (`Triangle`, `GWESP`, …)
prints the same table followed by a warning that the naive
pseudo-likelihood standard errors are suspect — refit with `method=:mcmle`
or use `se=:bootstrap` (see the estimation guide for the exact text).

### Accessing Results Programmatically

```julia
# Coefficient vector
coef(result)

# Standard errors
stderror(result)

# Variance-covariance matrix
vcov(result)

# Normal-theory confidence limits (one row per coefficient)
confint(result)

# The coefficient table itself, inspectable by index or term name
tbl = coeftable(result)
tbl["edges"].p_value
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
using Networks, ERGM
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

# Define and fit model. Closure is modelled with GWESP, not a raw
# Triangle count — see the note under this block.
terms = [
    Edges(),
    GWESP(0.5),
    NodeMatch(:group),
]

result = ergm(net, terms; method=:mple)
println(result)

# Simulate from fitted model: the simulated edge count sits near the
# observed 15 (inspect the mean), one diagnostic of model adequacy
sim_nets = simulate_ergm(result; n_sim=50, rng=Xoshiro(42))
println("Mean edges in simulations: ",
    mean(ne(s) for s in sim_nets))

# Goodness of fit
gof_result = gof(result; n_sim=50, stats=[:degree, :esp], rng=Xoshiro(43))
deg = only(s for s in gof_result.statistics if s.name == "degree")
println("Degree GOF p-values: ", deg.p_values)
```

!!! warning "Why not `Triangle()`? A degenerate fit, made visible"
    Replace `GWESP(0.5)` with `Triangle()` above and the MPLE still "converges" — but its simulations have **≈ 67 edges on average, against 15 observed** (of 105 dyads): the fitted model puts its mass near the complete graph, the textbook ERGM degeneracy of a raw triangle count. Always look at the simulated edge count after a fit; the estimation guide's [Handling Degeneracy](@ref "Handling Degeneracy") section explains how geometrically weighted terms can reduce this risk. They do not guarantee a nondegenerate fit.

    ```julia
    bad = ergm(net, [Edges(), Triangle(), NodeMatch(:group)]; method=:mple)
    mean(ne(s) for s in simulate_ergm(bad; n_sim=50, rng=Xoshiro(1)))   # ≈ 67, not 15
    ```

## Network Simulation

Simulate networks from a fitted model to assess whether the model reproduces key features of the observed network:

```julia
# Simulate 100 networks (burn-in and thinning default to the dyad-scaled
# rule 20 × n_dyads / max(100, n_dyads ÷ 10); pass burnin=/interval= to override)
sim_nets = simulate_ergm(result; n_sim=100, rng=Xoshiro(44))

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
gof_result = gof(result; n_sim=100, stats=[:degree, :esp, :distance], rng=Xoshiro(45))

# Degree distribution GOF (one GOFStatistic panel per statistic)
deg_gof = only(s for s in gof_result.statistics if s.name == "degree")
println("Observed degree distribution: ", deg_gof.observed)
println("Simulated mean: ", round.(vec(mean(deg_gof.simulated; dims=1)), digits=1))
```

## Best Practices

1. **Start simple**: Begin with `Edges()` only, then add terms incrementally
2. **Use MPLE for exploration**: Switch to MCMLE for final results
3. **Check convergence**: Verify `result.converged == true`
4. **Assess fit**: Use `gof()` to compare simulated and observed statistics; agreement does not establish that the model is correct
5. **Avoid degeneracy**: Use geometrically weighted terms (`GWESP`, `GWDegree`) instead of raw `Triangle` and `Kstar` for larger networks
6. **Set random seeds**: For reproducibility in MCMLE and simulation

## Next Steps

- Learn about all available [Model Terms](guide/terms.md)
- Understand [Model Estimation](guide/estimation.md) in detail
- Explore [Network Simulation](guide/simulation.md) from fitted models
- Run [Goodness-of-Fit](guide/diagnostics.md) diagnostics
