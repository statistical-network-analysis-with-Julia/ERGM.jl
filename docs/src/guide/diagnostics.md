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
seed, same result), `burnin`/`interval` (MCMC controls; they default to
the dyad-scaled rule every sampler shares, `20 × n_dyads` and
`max(100, n_dyads ÷ 10)` — see the [simulation guide](@ref "Choosing Burn-in and Interval")),
and `n_chains` (the simulations are split over independent chains run in
parallel, seeded deterministically from `rng`, so results are
thread-count-independent).

### Available Statistics

The `gof` function compares these network statistics:

| Statistic | Description | What it Captures |
|-----------|-------------|-----------------|
| `:degree` | Degree distribution | Overall connectivity patterns |
| `:idegree` / `:odegree` | In-/out-degree distributions | Direction-specific connectivity (directed networks) |
| `:esp` | Edgewise shared partner distribution | Local clustering / triadic closure |
| `:distance` | Geodesic distance distribution, plus an `"Inf"` level for unreachable pairs | Global network structure / reachability |

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

The distance GOF compares the distribution of shortest-path lengths over
the network's dyads — unordered pairs on an undirected network, ordered
pairs (out-distances) on a directed one, exactly R's `gof(..., GOF =
~distance)` — with one level per finite distance and a final `"Inf"` level
counting the pairs that cannot reach each other at all. That last row is
the reachability comparison, the one thing this panel adds over the degree
and ESP panels. On flomarriage the observed panel is `[20, 35, 32, 15, 3,
…, Inf: 15]` (the 15 pairs involving the isolated Pucci family), R's
`obs.dist`; the provenanced fixture pins it.

```julia
dist_gof = only(s for s in gof_result.statistics if s.name == "distance")

println("Geodesic distance GOF:")
for (d, label) in enumerate(dist_gof.labels)
    println("  Distance $label: obs=$(dist_gof.observed[d]), ",
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

## MCMLE Non-convergence

Before the MCMC diagnostics, check the estimation itself converged. An MCMLE
fit that exhausts `maxiter` warns at fit time (quoting the last max t-ratio,
Hotelling p-value and step length), prints the caveat under `Converged:
false` when shown, lists it in `Networks.approximations(result)`, and stores
the numbers in `result.mcmc_convergence`:

```julia
result = ergm(net, terms; method=:mcmle, n_samples=1000, rng=Xoshiro(1))
if !result.converged
    c = result.mcmc_convergence          # (iterations, step_length, t_ratios, hotelling_p, n_eff)
    println("max t-ratio ", maximum(c.t_ratios), ", Hotelling p ", c.hotelling_p,
            ", ESS ", c.n_eff, " after ", c.iterations, " iterations")
end
```

The same numbers are available for any sample through the `public`
`ERGM.mcmc_convergence(samples, targets)` (per-statistic t-ratios, the
Geyer effective sample size, the Hotelling T² p-value and the verdict). A
fit that converged has `step_length == 1.0` and every t-ratio below
`conv_threshold`. To continue an unconverged fit with a larger budget pass
`init=coef(result)`.

The standard errors of a converged MCMLE fit already include the
Monte-Carlo error of the estimate; `show` prints R's "MCMC %" column —
`round(100·(se − se_fisher)/se)`, the share of the *standard error* the
Monte-Carlo term adds — as `MCMC % of the standard error`, and the
Monte-Carlo part itself is `mcmc_se(result)`; see the
[estimation guide](@ref "Standard errors: Fisher information plus Monte-Carlo error").

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

The result is an `MCMCDiagnostics` whose `show` prints one row per term —
`println(diag)` is R's `mcmc.diagnostics(fit)` table:

```
MCMC diagnostics: 800 draws in 4 chains of 200, 200, 200, 200
                  lag-1 AC  ESS (lag-1)  ESS (Geyer)  Geweke z  Geweke p
edges                0.431        318.3        342.3     -0.98     0.329
gwesp.fixed.0.5      0.278        452.0        436.2      1.03     0.301
Prefer ESS (Geyer): it accounts for autocorrelation at every lag. A small Geweke p flags a chain that had not reached stationarity.
```

Its fields:

| Field | Description | Ideal Value |
|-------|-------------|-------------|
| `term_names` | Names of model terms | — |
| `autocorrelation` | Lag-1 autocorrelation per term (within chains) | Close to 0 |
| `effective_sample_size` | ESS from the lag-1 autocorrelation (most optimistic estimate) | > 100 |
| `ess_geyer` | Geyer initial-sequence ESS, using all lags (preferred) | > 100 |
| `geweke_z`, `geweke_p` | Geweke stationarity z-scores (first 10% vs last 50% of a chain) and two-sided p-values | `geweke_p` not small |
| `n_samples`, `chain_lengths` | Total MCMC samples and the per-chain counts | — |

With `n_chains > 1` every quantity is computed **within each chain** and
combined — the two ESS columns summed over chains (the Geyer column is then
exactly the `n_eff` of the MCMLE's own convergence test,
`result.mcmc_convergence.n_eff`), the Geweke test reported for the chain of
largest |z| — never across the seams of the concatenated sample.

Prefer `ess_geyer` over the lag-1 `effective_sample_size`: the
initial-sequence estimator accounts for autocorrelation at all lags. A
small `geweke_p` flags a chain that had not reached stationarity —
increase `burnin`.

### Interpreting Results

```julia
if !isnothing(result.mcmc_samples)
    diag = mcmc_diagnostics(result)
    println(diag)                                   # the per-term table
    low = diag.term_names[diag.ess_geyer .< 100]    # terms whose chain mixes poorly
    isempty(low) || println("Low ESS: ", join(low, ", "))
else
    println("No MCMC samples — model was fit with MPLE")
end
```

### Effective Sample Size

The lag-1 effective sample size (ESS) accounts for first-order
autocorrelation, chain by chain:

$$\text{ESS}_j = \sum_c L_c \cdot \frac{1 - \rho_{cj}}{1 + \rho_{cj}}$$

where $\rho_{cj}$ is the lag-1 autocorrelation of term $j$ in chain $c$ of
length $L_c$. The Geyer column replaces $(1-\rho)/(1+\rho)$ by the
initial-monotone-sequence estimate over all lags.

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
    println(mcmc_diagnostics(result))
end
```

## Computational Notes

- GOF computation time scales linearly with `n_sim`
- The simulations are split over `n_chains` chains; each chain's burn-in and thinning default to the dyad-scaled rule (`20 × n_dyads`, `max(100, n_dyads ÷ 10)`), so the cost of `gof` grows with the number of dyads
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
