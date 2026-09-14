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
| `StructuralTerm` | Network topology only | Edges, Triangle, GWESP, GWDSP, Degree |
| `NodalTerm` | Vertex attribute effects | NodeFactor, NodeMatch, NodeMix, NodeCov |
| `DyadicTerm` | Dyad-level covariates | EdgeCov |

!!! warning "Not implemented — what an R `ergm` user will not find"
    ERGM.jl implements the terms listed in this guide. What an R `ergm`
    user will not find, and what happens instead:

    - **Absent term families** — sender/receiver attribute terms `nodeicov`/`nodeocov`/`nodeifactor`/`nodeofactor`; the directed triadic terms `ttriple`/`ctriple`/`transitiveties`/`cyclicalties`/`asymmetric`; the `esp`/`dsp`/`nsp` count terms; `isolates` (only as `Degree(0)`), `balance`, `cycle`, `smalldiff`, `nodematch(keep=)`
      have no counterpart (there is no term to call, so nothing is
      silently mis-fit). Use `IStar`/`OStar` for directed stars,
      `Triangle` (= `ttriple + ctriple` on a directed network) for triadic
      closure, `GWESP`/`GWDSP` for shared-partner effects, and
      `NodeFactor`/`NodeCov` for a one-mode attribute main effect.

    - **Curved terms** — `gwesp(fixed=FALSE)`, `gwdegree(fixed=FALSE)`,
      `gwdsp(fixed=FALSE)` with the decay *estimated*. Every geometrically
      weighted term here takes a fixed decay (`GWESP(0.5)` is
      `gwesp(0.5, fixed=TRUE)`); there is no curved-family estimation.
    - **Bipartite terms and two-mode networks** — `b1degree`, `b2degree`,
      `b1factor`, `b2factor`, `b1nodematch`, `b1star`, …. `ERGMModel`
      (hence `fit_ergm`/`ergm`) refuses a network created with
      `bipartite=k` or a `BipartiteNetwork` with an `ArgumentError` naming
      the bipartite gap, rather than counting the impossible within-mode
      dyads as observations.
    - **`constraints=`** — `edges`, `degrees`, `blockdiag`, `observed`, …:
      a non-empty `constraints=` on `ERGMFormula` throws an `ArgumentError`.
    - **`offset()` terms** — no counterpart; every term is estimated.
    - **`nodecov`/`nodefactor`/… with an NA value** — refused at model
      construction with an `ArgumentError` naming the vertices without a
      value (statnet: "Attribute has missing data, which is not currently
      supported by ergm"). There is no zero-fill.
    - **Simulation and GOF under missing data** — a simulated network has
      a value at every dyad, so `simulate_ergm`/`gof` on a masked network
      only offer the warned `missing=:condition_on_face` (masked dyads
      frozen at face value) and refuse `missing=:mle`. Estimation under
      missing data is implemented: `mcmle(...; missing=:mle)` is R's
      missing-data MLE, and MPLE drops masked dyads.

## Structural Terms

These capture patterns in the network topology without reference to node attributes. The examples below assume the packages are loaded and a small example network exists:

```julia
using Networks, ERGM

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

### K-Star, OStar, IStar

The number of k-stars in the network — a central node connected to k others, $\sum_v \binom{d_v}{k}$ (statnet's `kstar(k)`). **Undirected networks only**, as in R ergm; a directed network gets the out-star and in-star terms `OStar(k)` / `IStar(k)` (statnet's `ostar(k)` / `istar(k)`, $\sum_v \binom{d^{out}_v}{k}$ and $\sum_v \binom{d^{in}_v}{k}$):

```julia
Kstar(2)   # Two-stars, "kstar2" — undirected networks only
Kstar(3)   # Three-stars

OStar(2)   # Out-two-stars, "ostar2" — directed networks only
IStar(2)   # In-two-stars,  "istar2" — directed networks only
```

`ERGMModel` refuses `Kstar` on a directed network with an `ArgumentError` naming `OStar`/`IStar` (R: "Term may not be used with networks with directed==TRUE"), and refuses `OStar`/`IStar` on an undirected one.

!!! warning "Changed in 0.2"
    Before 0.2 a directed network silently received *out*-stars under the label `kstar<k>`.

**Interpretation**: Controls the degree distribution. `Kstar(2)` captures variance in degree; `Kstar(3)` captures skewness. `OStar(2)` is activity spread (out-degree variance), `IStar(2)` popularity spread.

!!! warning "Degeneracy"
    Like `Triangle`, raw k-star terms can cause degeneracy. Use [`GWDegree`](@ref) (or `GWODegree`/`GWIDegree`) for a more stable alternative.

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

# decay = 0 — statnet's gwesp(0, fixed=TRUE), the most common applied
# specification: the number of edges with at least one shared partner
GWESP(0.0)
```

**Formula**: For each edge with ESP (edgewise shared partner) count $s$:

$$\text{GWESP} = \sum_{\text{edges}} e^\alpha \left(1 - (1 - e^{-\alpha})^s\right)$$

**Parameter α** (decay, ≥ 0): Controls how quickly additional shared partners are downweighted. Higher α = less downweighting; at α = 0 every edge with $s \ge 1$ contributes exactly 1.

**Coefficient names follow R's labels exactly**: an integer-valued decay is printed without a decimal point, so `GWESP(0.0)` is `"gwesp.fixed.0"` and `GWESP(1.0)` is `"gwesp.fixed.1"` (R's `gwesp(1, fixed=TRUE)`), while `GWESP(0.5)` is `"gwesp.fixed.0.5"`. A coefficient can therefore be compared with a statnet fit by name.

```julia
name(GWESP(0.0))   # "gwesp.fixed.0"
name(GWESP(1.0))   # "gwesp.fixed.1"
name(GWESP(0.5))   # "gwesp.fixed.0.5"
```

**Directed networks**: `type` selects the shared-partner definition for each directed edge $i \to j$, matching statnet's `dgwesp` types:

```julia
GWESP(0.5)               # :OTP — outgoing two-path i→k→j (statnet's directed default)
GWESP(0.5; type=:ITP)    # incoming two-path j→k→i
GWESP(0.5; type=:OSP)    # outgoing shared partner i→k←... i.e. i→k and j→k
GWESP(0.5; type=:ISP)    # incoming shared partner k→i and k→j
GWESP(0.5; type=:union)  # either-direction adjacency (pre-0.2 behavior,
                         # named "gwesp.union.fixed.<decay>"; not a statnet type)
```

The coefficient names follow statnet, **including the fact that R's label depends on the network's directedness**: on an undirected network the coefficient is `"gwesp.fixed.<decay>"` (`type` is ignored, in the label as in the statistic); on a *directed* network R ergm 4.12.0 names every type — the `:OTP` default included — `"gwesp.<type>.fixed.<decay>"`, so a directed fit prints `gwesp.OTP.fixed.0.5`, `gwesp.ITP.fixed.0.5`, … exactly as `coef()` of the statnet fit does. `ERGMModel`/`fit_ergm`, `summary_stats` and the two-argument `name(term, net)` all resolve the label against the network; the one-argument `name(GWESP(0.5))`, which cannot see a network, gives the undirected label for `:OTP`. The non-statnet `:union` is always `"gwesp.union.fixed.<decay>"`.

```julia
flo = load_dataset(:florentine_marriage)    # undirected
name(GWESP(0.5), flo)                       # "gwesp.fixed.0.5"
name(GWESP(0.5), net)                       # "gwesp.OTP.fixed.0.5" — `net` above is directed: R's label
ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), net).formula.terms.names   # ["edges", "gwesp.OTP.fixed.0.5"]
summary_stats(net, [GWESP(0.5)])            # (var"gwesp.OTP.fixed.0.5" = 0.0,)
```

!!! warning "Changed in 0.2"
    Directed `GWESP` previously counted either-direction shared partners while carrying statnet's OTP name. The default is now genuinely `:OTP`; use `type=:union` to reproduce the old statistic.

| Decay | Behavior |
|-------|----------|
| Small (0.1–0.3) | Strong downweighting — first shared partner matters most |
| Medium (0.5) | Moderate downweighting |
| Large (0.8–1.0) | Weak downweighting — approaches raw triangle count |

**Interpretation**: A positive coefficient indicates triadic closure, with diminishing returns for each additional shared partner. Preferred over `Triangle` for avoiding degeneracy.

### GWDegree (Geometrically Weighted Degree)

A geometrically downweighted summary of the degree distribution (statnet's `gwdegree(decay, fixed=TRUE)`, coefficient name `"gwdeg.fixed.<decay>"` — R's label):

```julia
# Default decay = 0.5
GWDegree()

# Custom decay
GWDegree(0.8)

# decay = 0: the number of non-isolated vertices
GWDegree(0.0)

name(GWDegree(0.5))   # "gwdeg.fixed.0.5"
name(GWDegree(1.0))   # "gwdeg.fixed.1"
```

**Formula**: For each vertex with degree $d$:

$$\text{GWDegree} = \sum_{\text{vertices}} e^\alpha \left(1 - (1 - e^{-\alpha})^d\right)$$

**Interpretation**: A positive coefficient indicates a preference for more even degree distributions (anti-preferential attachment). Preferred over `Kstar` for stability.

`GWDegree` is defined for **undirected networks only**, as in R ergm. On a
directed network `ERGMModel` refuses it with an `ArgumentError` naming the
directed variants, which model the in- and out-degree distributions
separately and match statnet's `gwidegree`/`gwodegree` (both with
`fixed=TRUE`):

```julia
GWIDegree(0.5)   # geometrically weighted in-degree, "gwideg.fixed.0.5"
GWODegree(0.5)   # geometrically weighted out-degree, "gwodeg.fixed.0.5"
GWIDegree(0.0)   # decay 0: the number of vertices with in-degree ≥ 1
```

Both are defined only for directed networks.

!!! warning "Changed in 0.2"
    Before 0.2 a directed network silently received the *out*-degree
    statistic under the undirected label, and the undirected coefficient was
    labelled `gwdegree.fixed.<decay>` where R writes `gwdeg.fixed.<decay>`.

### GWDSP (Geometrically Weighted Dyadwise Shared Partners)

The shared-partner analogue of `GWESP` summed over *all dyads* — tied or
not — instead of over edges only (statnet's `gwdsp(decay, fixed=TRUE)`):

```julia
GWDSP()                  # default decay = 0.5
GWDSP(0.8)               # custom decay
GWDSP(0.0)               # decay 0: the number of dyads with ≥ 1 shared partner
GWDSP(0.5; type=:OSP)    # directed shared-partner type, as for GWESP
```

For directed networks `type` selects the shared-partner definition
(`:OTP` outgoing two-path — the default, `:ITP`, `:OSP`, `:ISP`, or the
non-statnet `:union`), with statnet's `dgwdsp` conventions: **every type
is summed over all ordered dyads** (i, j), i ≠ j — the symmetric
`:OSP`/`:ISP` counts included, so each unordered pair contributes twice
(R's `ddsp` C code; on `samplike`, `gwdsp.OSP.fixed.0.5 = 316.085` and
`gwdsp.ISP.fixed.0.5 = 229.136`, pinned by the provenanced fixture). For
dyadwise shared partners `:OTP` and `:ITP` yield the same statistic. The
labels are direction-aware exactly as for `GWESP`: `"gwdsp.fixed.<decay>"`
on an undirected network, `"gwdsp.OTP.fixed.<decay>"` etc. on a directed
one.

!!! note "Changed in 0.2 (round 2)"
    `GWDSP(d; type=:OSP)` and `type=:ISP` on a directed network used to sum over *unordered* dyads and came out at exactly half of statnet's `dgwdsp` — with a consistently halved change statistic, so MPLE/MCMLE coefficients for these types were off by a factor of 2 relative to R. Both now follow R.

**Interpretation**: `GWESP` captures closure among *connected* dyads;
`GWDSP` captures the prevalence of two-paths regardless of whether the
dyad is tied. Including both separates "friends of friends" (open
two-paths) from actual triadic closure.

### Degree, IDegree, ODegree

The number of vertices with a given (in-/out-)degree — statnet's
`degree(d)`, `idegree(d)`, `odegree(d)`:

```julia
Degree(1)      # undirected networks only: count of degree-1 vertices
IDegree(0)     # directed only: count of in-degree-0 vertices
ODegree(2)     # directed only: count of out-degree-2 vertices
```

As in statnet, `Degree` is only defined for undirected networks — use
`IDegree`/`ODegree` on directed networks. A vector or range of degrees is
**one term that expands into one statistic per degree** when the model is
built (exactly like a multi-level `NodeFactor`), so the statnet formula
`~ edges + degree(0:2)` becomes:

```julia
flo = load_dataset(:florentine_marriage)
fit = fit_ergm(flo, [Edges(), Degree(0:2)])
fit.model.formula.terms.names          # ["edges", "degree0", "degree1", "degree2"]

summary_stats(flo, [Edges(), Degree(0:2)])
# (edges = 20.0, degree0 = 1.0, degree1 = 4.0, degree2 = 2.0)
```

The unexpanded `Degree(0:2)` is a specification, not a statistic:
`compute(Degree(0:2), net)` throws an `ArgumentError` saying so, and
`name(Degree(0:2))` is the placeholder `"degree(0,1,2)"`.

**Interpretation**: Degree-count terms model specific features of the
degree distribution exactly (e.g. `Degree(0)` for the number of
isolates). For a smooth overall degree effect prefer the geometrically
weighted terms.

## Nodal Attribute Terms

These incorporate vertex-level attributes for modeling homophily and covariate effects.

### NodeFactor

Main effect of a categorical vertex attribute (R ergm's `nodefactor`): for each included attribute level, the number of times a vertex with that level appears as an endpoint of an edge (edges within the level count twice).

```julia
# One statistic per level, first (sorted) level dropped as the reference
# — exactly like R's nodefactor("gender")
NodeFactor(:gender)

# Keep all levels (statnet's base=0 / levels=TRUE)
NodeFactor(:gender; base=0)

# Choose the included levels explicitly, in order
NodeFactor(:gender; levels=["F"])

# A single-level term directly (no expansion)
NodeFactor(:gender; level="F")
```

The levels are resolved from the network at model construction, where the
multi-level term expands into per-level statistics named
`"nodefactor.<attr>.<level>"`.

!!! warning "Changed in 0.2"
    `NodeFactor(attr)` previously produced a *single* statistic counting
    the endpoint appearances of every vertex with the attribute — which is
    collinear with `Edges` by construction (every edge has two endpoints),
    so models with both were unidentified. It now matches statnet: one
    statistic per attribute level, with the first (sorted) level dropped
    as the reference category. Pass `base=0` to keep all levels.

**Interpretation**: A positive coefficient for level "F" means that female actors form more ties than actors in the reference category.

### NodeCov

Main effect of a continuous vertex attribute. Sums the attribute values of both endpoints for each edge.

The attribute must be set on **every** vertex — for this and every other attribute-based term. As in statnet (which refuses NA attribute values), `ERGMModel` throws an `ArgumentError` naming the term, the attribute, how many vertices lack a value and which ones; before 0.2 those vertices were silently zero-filled into the design.

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

### NodeMix

Mixing-matrix cell counts for a categorical vertex attribute (R ergm's `nodemix`): one statistic per mixing cell, counting the edges whose endpoint levels fall in that cell.

```julia
# One statistic per mixing cell, first cell dropped as the reference
# (statnet's levels2=-1 default)
NodeMix(:gender)

# Keep every cell
NodeMix(:gender; levels2=0)

# Select cells by index into the statnet-ordered cell list
NodeMix(:gender; levels2=[2, 3])

# A single cell directly (tail level, head level for directed networks)
NodeMix(:gender, "F", "M")
```

Cells follow statnet's ordering over the sorted levels — unordered pairs
`(u_i, u_j)`, `i ≤ j`, in column-major order for undirected networks, all
ordered (tail, head) pairs for directed ones — and the statistics are
named `"mix.<attr>.<l1>.<l2>"`. Like `NodeFactor`, the multi-cell term
expands into per-cell statistics at model construction.

**Interpretation**: Each coefficient measures the propensity of ties in
that particular level combination relative to the reference cell — a full
mixing structure, generalizing `NodeMatch`/`NodeMismatch`.

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
using Networks, ERGM

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
# Named summary of network statistics (the page's 3-node directed example:
# on a DIRECTED network the gwesp label carries R's type, gwesp.OTP.fixed.0.5)
stats = summary_stats(net, terms)
println(stats)
# (edges = 1.0, var"gwesp.OTP.fixed.0.5" = 0.0, var"nodematch.gender" = 0.0, var"nodecov.age" = 55.0, var"absdiff.age" = 5.0)
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
| Are two-paths prevalent (tied or not)? | GWDSP |
| Is the degree distribution uneven? | Kstar, GWDegree (undirected); OStar, IStar, GWODegree, GWIDegree (directed) |
| Are specific degrees over-represented (e.g. isolates)? | Degree, IDegree, ODegree |
| Is there attribute homophily? | NodeMatch, AbsDiff |
| Do attributes affect sociality? | NodeFactor, NodeCov |
| Is there a full mixing structure between groups? | NodeMix |
| Do spatial/dyadic factors matter? | EdgeCov |

### Best Practices

1. **Always include Edges**: It controls baseline density and is needed in every model
2. **Prefer geometrically weighted terms**: Use `GWESP` over `Triangle` and `GWDegree` over `Kstar` to avoid degeneracy
3. **Start simple**: Build up from `Edges()` to more complex specifications
4. **Check for multicollinearity**: Highly correlated terms can inflate standard errors
5. **Use change statistics for validation**: Verify `change_stat` is consistent with `compute` on small networks
6. **Consider network size**: Some terms (Triangle, Kstar) become problematic at larger network sizes
