# Terms API Reference

This page documents all ERGM terms available in ERGM.jl.

## Structural Terms

Terms based purely on network structure.

```@docs
Edges
Mutual
Triangle
Kstar
TwoPath
GWESP
GWDSP
GWDegree
GWIDegree
GWODegree
Degree
IDegree
ODegree
```

## Nodal Attribute Terms

Terms incorporating vertex-level attributes.

```@docs
NodeFactor
NodeCov
NodeMatch
NodeMismatch
NodeMix
AbsDiff
```

## Dyadic Terms

Terms based on dyad-level covariates.

```@docs
EdgeCov
```

## Term Traits

The public protocol by which a term — ERGM.jl's own or a third party's —
declares what it needs and where it belongs. ERGM.jl acts on the declarations
when an [`ERGMModel`](@ref) is constructed: a term whose declared vertex/edge
attributes are absent from the network, or whose direction requirement the
network does not meet, is rejected with an `ArgumentError` instead of silently
contributing an all-zero design column.

```@docs
required_vertex_attributes
required_edge_attributes
requires_directed
requires_undirected
```

Two further traits complete the protocol:

- [`is_dyad_dependent`](@ref) — whether the change statistic reads other dyads
  (conservative default `true`); it drives the pseudo-likelihood caveat and the
  MCMLE bridge reference.
- `Networks.supports_missing` — whether the term's statistic honours the
  missing-dyad mask, i.e. is invariant to the face value of a dyad marked with
  `set_missing_dyad!` (default `false`, the honest answer for every built-in
  term: they count masked dyads at face value, and the principled missing-data
  treatment lives in the estimator — see [`mple`](@ref)).

A third-party term declares them by adding methods:

```julia
using ERGM
using Networks

struct RichClub <: AbstractERGMTerm
    attr::Symbol
end

ERGM.name(t::RichClub) = "richclub.$(t.attr)"
# ... plus compute(t, net) and change_stat(t, net, i, j)

ERGM.required_vertex_attributes(t::RichClub) = (t.attr,)
ERGM.requires_directed(::RichClub)           = true
ERGM.is_dyad_dependent(::RichClub)           = true
Networks.supports_missing(::RichClub)        = false
```

`ERGMUserterms.validate_term` exercises every one of these declarations against
a network; `ERGMUserterms.jl/examples/MyTermPackage` is a complete package
template built on them.

!!! note "Pre-0.2 private names"
    The traits were internal before v0.2 (`ERGM._vertex_attribute`,
    `ERGM._requires_directed`, `ERGM._requires_undirected`). The private names
    remain as aliases of the same generics, so downstream methods declared
    against them (TERGM.jl's `ERGM._requires_directed(::Delrecip) = true`) keep
    working, but new code should use the public names.
