using LocalErgotropy
using Documenter

DocMeta.setdocmeta!(LocalErgotropy, :DocTestSetup, :(using LocalErgotropy); recursive=true)

makedocs(;
    modules=[LocalErgotropy],
    authors="Federico Cerisola",
    sitename="LocalErgotropy.jl",
    format=Documenter.HTML(;
        canonical="https://cerisola.github.io/LocalErgotropy.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cerisola/LocalErgotropy.jl",
    devbranch="main",
)
