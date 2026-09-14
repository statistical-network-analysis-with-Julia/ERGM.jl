# Types API Reference

```@meta
CurrentModule = ERGM
```

This page documents the core data types in ERGM.jl.

## Abstract Term Types

### AbstractERGMTerm

```@docs
AbstractERGMTerm
```

### StructuralTerm

```@docs
StructuralTerm
```

### NodalTerm

```@docs
NodalTerm
```

### DyadicTerm

```@docs
DyadicTerm
```

### ConstraintTerm

```@docs
ConstraintTerm
```

## Term Collections

### TermSet

```@docs
TermSet
```

## Model Specification

### ERGMFormula

```@docs
ERGMFormula
```

### ERGMModel

```@docs
ERGMModel
```

## Results

### ERGMResult

```@docs
ERGMResult
```

### MCMLEConvergence

```@docs
ERGM.MCMLEConvergence
```

### MCMCDiagnostics

```@docs
MCMCDiagnostics
```

## Interface Functions

### compute

```@docs
compute
```

### change_stat

```@docs
change_stat
```

### name

```@docs
name
```

## Batch Operations

```@docs
compute_all
change_stat_all
summary_stats
```

## Result Accessors

The full ecosystem StatsAPI surface (`coef`, `stderror`, `vcov`, `confint`,
`loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `coeftable`) is defined on
`ERGMResult`; `Networks.check_statsapi(fit; strict=true)` passes.

```@docs
coef(::ERGMResult)
stderror(::ERGMResult)
vcov(::ERGMResult)
confint(::ERGMResult)
coeftable(::ERGMResult)
```
