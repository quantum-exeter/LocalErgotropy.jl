using LocalErgotropy
using Documenter

DocMeta.setdocmeta!(LocalErgotropy, :DocTestSetup, :(using LocalErgotropy); recursive=true)

makedocs(;
    modules=[LocalErgotropy],
    authors="Federico Cerisola <federico@cerisola.net>",
    sitename="LocalErgotropy.jl",
    format=Documenter.HTML(;
        canonical="https://quantum-exeter.github.io/LocalErgotropy.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/quantum-exeter/LocalErgotropy.jl",
    devbranch="main",
)
