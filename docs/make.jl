using Documenter
using ERGM

DocMeta.setdocmeta!(ERGM, :DocTestSetup, :(using ERGM); recursive=true)

makedocs(
    sitename = "ERGM.jl",
    modules = [ERGM],
    authors = "Statistical Network Analysis with Julia",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://Statistical-network-analysis-with-Julia.github.io/ERGM.jl",
        edit_link = "main",
    ),
    repo = "https://github.com/Statistical-network-analysis-with-Julia/ERGM.jl/blob/{commit}{path}#{line}",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Model Terms" => "guide/terms.md",
            "Model Estimation" => "guide/estimation.md",
            "Network Simulation" => "guide/simulation.md",
            "Goodness of Fit" => "guide/diagnostics.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Terms" => "api/terms.md",
            "Estimation" => "api/estimation.md",
        ],
    ],
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(
    repo = "github.com/Statistical-network-analysis-with-Julia/ERGM.jl.git",
    devbranch = "main",
    versions = [
        "stable" => "dev", # serve dev docs at /stable until a release is tagged
        "dev" => "dev",
    ],
    push_preview = true,
)
