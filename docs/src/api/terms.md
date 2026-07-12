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
