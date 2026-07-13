# Goodness of Fit and Diagnostics

ERGM.jl provides tools for assessing how well a fitted model reproduces key properties of the observed network. Goodness-of-fit (GOF) diagnostics compare observed network statistics to their distributions under the fitted model, while MCMC diagnostics assess convergence of the MCMLE estimation.

## Why Diagnostics?

Even a model that converges successfully may not capture important features of the observed network. GOF diagnostics help answer:

- Does the model reproduce the observed degree distribution?
- Does it capture the right amount of local clustering (shared partners)?
- Does it get the geodesic distance distribution right?
- Are there systematic departures that suggest missing terms?

## Goodness-of-Fit Assessment

### Running GOF

```julia
using Networks, ERGM
using Random
using Statistics

Random.seed!(42)

# Example network: Florentine marriage ties with a categorical attribute
net = load_dataset(:florentine_marriage)
set_vertex_attribute!(net, :gender,
    Dict(v => (isodd(v) ? "F" : "M") for v in 1:nv(net)))

# Fit a model
terms = [Edges(), GWESP(0.5), NodeMatch(:gender)]
result = ergm(net, terms; method=:mple)

# Run goodness-of-fit with 100 simulated networks
gof_result = gof(result; n_sim=100, stats=[:degree, :esp, :distance])
```

`gof` also accepts `rng` (all simulation randomness flows from it — same
seed, same result), `burnin`/`interval` (MCMC controls), and `n_chains`
(the simulations are split over independent chains run in parallel,
seeded deterministically from `rng`, so results are thread-count-independent).

### Available Statistics

The `gof` function compares these network statistics:

| Statistic | Description | What it Captures |
|-----------|-------------|-----------------|
| `:degree` | Degree distribution | Overall connectivity patterns |
| `:idegree` / `:odegree` | In-/out-degree distributions | Direction-specific connectivity (directed networks) |
| `:esp` | Edgewise shared partner distribution | Local clustering / triadic closure |
| `:distance` | Geodesic distance distribution | Global network structure / reachability |

For **directed** networks, requesting `:degree` produces separate
`idegree` and `odegree` panels (as R ergm's GOF does), and the `:esp`
panel uses the `esp_type` shared-partner definition (default `:OTP`,
statnet's directed default — same types as `GWESP`).

### GOF Results

`gof` returns a `Networks.GOFResult` — the goodness-of-fit container
shared by every model package in the ecosystem. Its `statistics` field
holds one `GOFStatistic` per panel:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `String` | Panel name (`"degree"`, `"esp"`, ...) |
| `labels` | `Vector{String}` | Level labels (degree 0, 1, 2, ...) |
| `observed` | `Vector{Float64}` | Observed distribution |
| `simulated` | `Matrix{Float64}` | One row per simulated network |
| `p_values` | `Vector{Float64}` | Two-sided Monte-Carlo p-values per level |

p-values use the shared `(1 + k)/(N + 1)` Monte-Carlo estimator
(`Networks.mc_pvalue`), so they are never exactly zero. `println(gof_result)`
renders observed value, simulation envelope, and p-value per level.

### Interpreting GOF Results

```julia
using Statistics

# Degree distribution panel
deg_gof = only(s for s in gof_result.statistics if s.name == "degree")

println("Degree distribution GOF:")
for d in 0:min(10, length(deg_gof.observed)-1)
    obs = deg_gof.observed[d+1]
    sim_mean = round(mean(deg_gof.simulated[:, d+1]), digits=1)
    sim_sd = round(std(deg_gof.simulated[:, d+1]), digits=1)
    p = round(deg_gof.p_values[d+1], digits=3)
    println("  Degree $d: obs=$obs, sim=$sim_mean ± $sim_sd, p=$p")
end
```

**Interpreting p-values**:

- p close to 0.5: Model reproduces this feature well
- p close to 0.0 or 1.0: Systematic deviation — model under/over-predicts
- p < 0.05 or p > 0.95: Significant departure from the model

### Degree Distribution

The degree GOF compares the count of nodes with each degree value:

```julia
deg_gof = only(s for s in gof_result.statistics if s.name == "degree")

# Good fit: observed values fall within simulated range
for d in 0:length(deg_gof.observed)-1
    sim_col = deg_gof.simulated[:, d+1]
    within_range = abs(deg_gof.observed[d+1] - mean(sim_col)) < 2 * std(sim_col)
    status = within_range ? "OK" : "POOR"
    println("Degree $d: $status")
end
```

### Edgewise Shared Partners

The ESP GOF compares how many edges have 0, 1, 2, ... shared partners:

```julia
esp_gof = only(s for s in gof_result.statistics if s.name == "esp")

println("ESP distribution GOF:")
for e in 0:length(esp_gof.observed)-1
    println("  ESP $e: obs=$(esp_gof.observed[e+1]), ",
            "sim=$(round(mean(esp_gof.simulated[:, e+1]), digits=1))")
end
```

### Geodesic Distance

The distance GOF compares the distribution of shortest path lengths between all reachable pairs:

```julia
dist_gof = only(s for s in gof_result.statistics if s.name == "distance")

println("Geodesic distance GOF:")
for d in 1:length(dist_gof.observed)
    println("  Distance $d: obs=$(dist_gof.observed[d]), ",
            "sim=$(round(mean(dist_gof.simulated[:, d]), digits=1))")
end
```

## Diagnosing Poor Fit

### What to Do When GOF Fails

| Poor GOF on | Likely Missing | Suggested Terms |
|-------------|----------------|-----------------|
| Degree distribution | Degree heterogeneity control | `GWDegree`, `NodeCov`, `NodeFactor` |
| ESP distribution | Triadic closure | `GWESP`, `Triangle` |
| Distance distribution | Global connectivity | `Edges` (adjust), consider network constraints |

### Iterative Model Building

```julia
# Model 1: Edges only
r1 = ergm(net, [Edges()])
g1 = gof(r1; n_sim=50)

# Model 2: Add triadic closure
r2 = ergm(net, [Edges(), GWESP(0.5)])
g2 = gof(r2; n_sim=50)

# Model 3: Add attribute effects
r3 = ergm(net, [Edges(), GWESP(0.5), NodeMatch(:gender)])
g3 = gof(r3; n_sim=50)

# Compare degree GOF across models
for (i, g) in enumerate([g1, g2, g3])
    deg = only(s for s in g.statistics if s.name == "degree")
    println("Model $i degree GOF mean p-value: $(round(mean(deg.p_values), digits=3))")
end
```

## MCMC Diagnostics

For models fit with MCMLE, MCMC diagnostics assess whether the MCMC sampler has converged and mixed well.

### Running MCMC Diagnostics

```julia
# Fit with MCMLE
result = ergm(net, terms; method=:mcmle, verbose=true)

# Get MCMC diagnostics
diag = mcmc_diagnostics(result)
```

`mcmc_diagnostics` requires a fit with MCMC samples. Calling it on an
MPLE fit throws an `ArgumentError` (MPLE draws no MCMC sample, so there
is nothing to diagnose) — refit with `method=:mcmle` first.

### Understanding MCMC Diagnostics

The diagnostics return:

| Field | Description | Ideal Value |
|-------|-------------|-------------|
| `term_names` | Names of model terms | — |
| `autocorrelation` | Lag-1 autocorrelation per term | Close to 0 |
| `effective_sample_size` | ESS from the lag-1 autocorrelation (most optimistic estimate) | > 100 |
| `ess_geyer` | Geyer initial-sequence ESS, using all lags (preferred) | > 100 |
| `geweke_z`, `geweke_p` | Geweke stationarity z-scores (first 10% vs last 50%) and two-sided p-values | `geweke_p` not small |
| `n_samples` | Total MCMC samples | — |

Prefer `ess_geyer` over the lag-1 `effective_sample_size`: the
initial-sequence estimator accounts for autocorrelation at all lags. A
small `geweke_p` flags a chain that had not reached stationarity —
increase `burnin`.

### Interpreting Results

```julia
if !isnothing(result.mcmc_samples)
    diag = mcmc_diagnostics(result)

    println("MCMC Diagnostics:")
    println("-"^50)
    for (i, tname) in enumerate(diag.term_names)
        ac = round(diag.autocorrelation[i], digits=3)
        ess = round(diag.effective_sample_size[i], digits=0)
        status = ess > 100 ? "OK" : "LOW"
        println("  $(rpad(tname, 20)) AC=$ac  ESS=$ess  [$status]")
    end
else
    println("No MCMC samples — model was fit with MPLE")
end
```

### Effective Sample Size

The effective sample size (ESS) accounts for autocorrelation:

$$\text{ESS}_j = n \cdot \frac{1 - \rho_j}{1 + \rho_j}$$

Where $\rho_j$ is the lag-1 autocorrelation for term $j$.

| ESS | Assessment |
|-----|-----------|
| > 200 | Good mixing |
| 100–200 | Acceptable |
| < 100 | Poor mixing — increase samples or interval |

### Improving MCMC Mixing

If diagnostics indicate poor mixing:

<!-- skip-check -->
```julia
# Increase samples and thinning
result = ergm(net, terms;
    method = :mcmle,
    n_samples = 5000,      # More samples
    burnin = 5000,          # Longer burn-in
    interval = 500,         # More thinning
    verbose = true
)
```

## Complete Diagnostic Workflow

```julia
using Networks, ERGM
using Random

Random.seed!(42)

# Fit model
terms = [Edges(), GWESP(0.5), GWDegree(0.5), NodeMatch(:gender)]
result = ergm(net, terms; method=:mple)

# 1. Check convergence
println("Converged: ", result.converged)

# 2. Run GOF
gof_result = gof(result; n_sim=100, stats=[:degree, :esp, :distance])

# 3. Summarize GOF
for stat in gof_result.statistics
    mean_p = mean(stat.p_values)
    println("$(stat.name) GOF (mean p): $(round(mean_p, digits=3))")
end

# 4. MCMC diagnostics (if MCMLE)
if !isnothing(result.mcmc_samples)
    diag = mcmc_diagnostics(result)
    println("\nMCMC ESS: ", round.(diag.effective_sample_size, digits=0))
end
```

## Computational Notes

- GOF computation time scales linearly with `n_sim`
- Each simulation runs its own MCMC chain (with default burn-in and interval)
- Distance computation uses Graphs.jl's `gdistances` and can be slow for very large networks
- For quick diagnostics, use `n_sim=50`; for publication, use `n_sim=200+`

## Best Practices

1. **Always run GOF**: A converged model can still fit poorly
2. **Check all three statistics**: Degree, ESP, and distance capture different aspects
3. **Use enough simulations**: At least 50 for exploration, 100+ for final assessment
4. **Iterate on model specification**: Use GOF results to guide term selection
5. **Compare models**: Run GOF on multiple specifications to find the best fit
6. **Check MCMC convergence**: For MCMLE, verify adequate ESS before trusting results
7. **Set random seeds**: For reproducible diagnostic results
