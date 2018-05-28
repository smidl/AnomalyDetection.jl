_ARGS = ARGS
include("generate_tex_tables.jl")

ARGS = _ARGS
include("generate_tex_graphs.jl")

