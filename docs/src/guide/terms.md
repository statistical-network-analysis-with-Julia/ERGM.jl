# Model Terms

ERGM terms are the building blocks of an ERGM. Each term computes a network statistic that captures a specific structural mechanism. All terms implement a common interface and can be freely combined in models.

## Term Interface

All terms implement three methods:

<!-- skip-check -->
```julia
compute(term, net) -> Float64
change_stat(term, net, i, j) -> Float64
name(term) -> String
```

The `compute` function calculates the full statistic value for the network. The `change_stat` function computes the *add-direction* change statistic $g(y^+_{ij}) - g(y^-_{ij})$: the statistic with edge $(i, j)$ present minus the statistic with it absent, holding the rest of the network fixed. Its value does not depend on whether the edge currently exists — this state-independent quantity is what the MPLE design matrix and the Metropolis–Hastings sampler both require (the sampler negates it for removal proposals).

## Term Categories

ERGM.jl organizes terms into three categories:

| Type | Description | Examples |
|------|-------------|----------|
| `StructuralTerm` | Network topology only | Edges, Triangle, GWESP |
| `NodalTerm` | Vertex attribute effects | NodeMatch, NodeCov |
| `DyadicTerm` | Dyad-level covariates | EdgeCov |

## Structural Terms

These capture patterns in the network topology without reference to node attributes. The examples below assume the packages are loaded and a small example network exists:

```julia
using Network, ERGM

net = network(3; directed=true)
add_edge!(net, 1, 2)
set_edge_attribute!(net, :distance, 1, 2, 1.5)
```

### Edges

The most basic ERGM term — the total number of edges in the network. Analogous to an intercept in regression.

```julia
Edges()
```

**Interpretation**: Controls baseline network density. A negative coefficient (common) indicates sparse networks are more likely than dense ones.

### Mutual

The number of reciprocated dyads (directed networks only):

```julia
Mutual()
```

**Interpretation**: A positive coefficient indicates a tendency toward reciprocity — if $i$ sends a tie to $j$, then $j$ is more likely to send a tie to $i$.

### Triangle

The number of triangles in the network:

```julia
Triangle()
```

Visual representation:

```text
  k
 / \
i — j   ← all three edges present
```

**Interpretation**: A positive coefficient indicates triadic closure — "friends of friends become friends."

!!! warning "Degeneracy"
    The `Triangle` term can cause model degeneracy in larger networks (the model places nearly all probability on either empty or complete networks). Use [`GWESP`](@ref) instead for robust triangle effects.

### K-Star

The number of k-stars in the network. A k-star consists of a central node connected to k other nodes.

```julia
Kstar(2)   # Two-stars
Kstar(3)   # Three-stars
```

**Interpretation**: Controls the degree distribution. `Kstar(2)` captures variance in degree; `Kstar(3)` captures skewness.

!!! warning "Degeneracy"
    Like `Triangle`, raw k-star terms can cause degeneracy. Use [`GWDegree`](@ref) for a more stable alternative.

### TwoPath

The number of two-paths. For directed networks this counts pairs of edges $h \to v$, $v \to k$ with $h \neq k$ (statnet's `twopath`); for undirected networks it counts 2-stars, $\sum_v \binom{d_v}{2}$:

```julia
TwoPath()
```

**Interpretation**: Captures the tendency for directed paths — nodes that receive ties also tend to send ties (to other nodes).

### GWESP (Geometrically Weighted Edgewise Shared Partners)

A geometrically downweighted count of shared partners across edges:

```julia
# Default decay = 0.5
GWESP()

# Custom decay parameter
GWESP(0.8)
```

**Formula**: For each edge with ESP (edgewise shared partner) count $s$:

$$\text{GWESP} = \sum_{\text{edges}} e^\alpha \left(1 - (1 - e^{-\alpha})^s\right)$$

**Parameter α** (decay): Controls how quickly additional shared partners are downweighted. Higher α = less downweighting.

**Directed networks**: `type` selects the shared-partner definition for each directed edge $i \to j$, matching statnet's `dgwesp` types:

```julia
GWESP(0.5)               # :OTP — outgoing two-path i→k→j (statnet's directed default)
GWESP(0.5; type=:ITP)    # incoming two-path j→k→i
GWESP(0.5; type=:OSP)    # outgoing shared partner i→k←... i.e. i→k and j→k
GWESP(0.5; type=:ISP)    # incoming shared partner k→i and k→j
GWESP(0.5; type=:union)  # either-direction adjacency (pre-0.2 behavior,
                         # named "gwesp.union.fixed.<decay>"; not a statnet type)
```

The coefficient names follow statnet: `"gwesp.fixed.<decay>"` for `:OTP` (and for undirected networks, where `type` is ignored), `"gwesp.ITP.fixed.<decay>"` etc. for the other types.

!!! warning "Changed in 0.2"
    Directed `GWESP` previously counted either-direction shared partners while carrying statnet's OTP name. The default is now genuinely `:OTP`; use `type=:union` to reproduce the old statistic.

| Decay | Behavior |
|-------|----------|
| Small (0.1–0.3) | Strong downweighting — first shared partner matters most |
| Medium (0.5) | Moderate downweighting |
| Large (0.8–1.0) | Weak downweighting — approaches raw triangle count |

**Interpretation**: A positive coefficient indicates triadic closure, with diminishing returns for each additional shared partner. Preferred over `Triangle` for avoiding degeneracy.

### GWDegree (Geometrically Weighted Degree)

A geometrically downweighted summary of the degree distribution:

```julia
# Default decay = 0.5
GWDegree()

# Custom decay
GWDegree(0.8)
```

**Formula**: For each vertex with degree $d$:

$$\text{GWDegree} = \sum_{\text{vertices}} e^\alpha \left(1 - (1 - e^{-\alpha})^d\right)$$

**Interpretation**: A positive coefficient indicates a preference for more even degree distributions (anti-preferential attachment). Preferred over `Kstar` for stability.

## Nodal Attribute Terms

These incorporate vertex-level attributes for modeling homophily and covariate effects.

### NodeFactor

Main effect of a categorical vertex attribute. Counts edges incident to nodes with a specific attribute value.

```julia
# Effect of all levels
NodeFactor(:gender)

# Effect of a specific level
NodeFactor(:gender; level="F")
```

**Interpretation**: A positive coefficient for level "F" means that female actors form more ties than expected by chance.

### NodeCov

Main effect of a continuous vertex attribute. Sums the attribute values of both endpoints for each edge.

```julia
# Basic continuous effect
NodeCov(:age)

# With transform
NodeCov(:age; transform=:log)    # log(value + 1)
NodeCov(:age; transform=:sqrt)   # sqrt(value)
```

**Interpretation**: A positive coefficient means nodes with higher attribute values are more likely to form ties.

### NodeMatch

Homophily effect — tendency for edges between vertices with matching attribute values:

```julia
# Uniform homophily: one statistic counting all matched edges
NodeMatch(:gender)

# Differential homophily (R's nodematch(diff=TRUE)): one statistic per
# level, each counting only that level's matched edges. The term system is
# one-statistic-per-term, so construct one term per level:
[NodeMatch(:gender; diff=true, level=l) for l in ("F", "M")]
```

Statistic names follow R ergm: `"nodematch.gender"` and `"nodematch.gender.F"` / `"nodematch.gender.M"`.

**Interpretation**:

- `NodeMatch` > 0: Homophily — actors prefer same-type partners
- `NodeMatch` < 0: Heterophily — actors prefer different-type partners
- With `diff=true`, each level's coefficient measures within-level homophily separately

!!! warning "Changed in 0.2"
    `NodeMatch(attr; diff=true)` previously counted *mismatched* edges — a different model from R's `nodematch(diff=TRUE)` under the same keyword. `diff=true` now means differential homophily as in R (and requires `level`); the mismatch count is available as `NodeMismatch`.

### NodeMismatch

Heterophily effect — the number of edges whose endpoints have *different* attribute values (this package's pre-0.2 `NodeMatch(...; diff=true)` statistic; it has no R `nodematch` equivalent):

```julia
NodeMismatch(:gender)
```

**Interpretation**: A positive coefficient means actors prefer different-type partners.

### AbsDiff

Absolute difference in a continuous attribute between edge endpoints:

```julia
# Linear absolute difference
AbsDiff(:age)

# Powered absolute difference
AbsDiff(:age; pow=2.0)   # |age_i - age_j|²
```

**Interpretation**:

- `AbsDiff` < 0: Actors with similar attribute values are more likely to form ties
- `AbsDiff` > 0: Actors with different attribute values are more likely to form ties

## Dyadic Terms

These incorporate dyad-level covariates.

### EdgeCov

Dyad-level covariate effect. Takes an $n \times n$ matrix where entry $(i,j)$ is the covariate value for that dyad.

```julia
# From a covariate matrix
distance_matrix = [0.0 1.5 3.0; 1.5 0.0 2.0; 3.0 2.0 0.0]
EdgeCov(distance_matrix; name="distance")

# From an edge attribute stored in the network
EdgeCov(net, :distance)
```

**Interpretation**: A negative coefficient on distance means that geographically close nodes are more likely to form ties.

## Using Terms in Practice

### Building a Model

```julia
using Network, ERGM

# Set attributes
set_vertex_attribute!(net, :gender, Dict(1=>"M", 2=>"F", 3=>"M"))
set_vertex_attribute!(net, :age, Dict(1=>25.0, 2=>30.0, 3=>28.0))

# Build comprehensive model
terms = [
    # Structural
    Edges(),
    GWESP(0.5),

    # Attribute effects
    NodeMatch(:gender),
    NodeCov(:age),
    AbsDiff(:age),
]
```

### Computing Statistics Manually

```julia
# Compute individual term
edges_term = Edges()
println("Edge count: ", compute(edges_term, net))

# Compute change statistic
println("Change from adding (1,2): ", change_stat(edges_term, net, 1, 2))

# Compute all terms at once via TermSet
ts = TermSet(terms)
stats = compute_all(ts, net)
println("All statistics: ", stats)
```

### Summary Statistics

```julia
# Named summary of network statistics
stats = summary_stats(net, terms)
println(stats)
# (edges = 12.0, gwesp.fixed.0.5 = 4.3, nodematch.gender = 7.0, ...)
```

### Change Statistics for All Terms

```julia
# How all statistics change when adding edge (1, 2)
ts = TermSet(terms)
deltas = change_stat_all(ts, net, 1, 2)
println("Change statistics: ", deltas)
```

## Choosing Terms

### By Research Question

| Question | Terms |
|----------|-------|
| What is the baseline density? | Edges |
| Is there reciprocity? | Mutual |
| Does the network cluster? | Triangle, GWESP |
| Is the degree distribution uneven? | Kstar, GWDegree |
| Is there attribute homophily? | NodeMatch, AbsDiff |
| Do attributes affect sociality? | NodeFactor, NodeCov |
| Do spatial/dyadic factors matter? | EdgeCov |

### Best Practices

1. **Always include Edges**: It controls baseline density and is needed in every model
2. **Prefer geometrically weighted terms**: Use `GWESP` over `Triangle` and `GWDegree` over `Kstar` to avoid degeneracy
3. **Start simple**: Build up from `Edges()` to more complex specifications
4. **Check for multicollinearity**: Highly correlated terms can inflate standard errors
5. **Use change statistics for validation**: Verify `change_stat` is consistent with `compute` on small networks
6. **Consider network size**: Some terms (Triangle, Kstar) become problematic at larger network sizes
