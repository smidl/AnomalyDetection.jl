module Experiments

using JLD
push!(LOAD_PATH, "../src")
using AnomalyDetection
using Flux
using Distances

# paths
# SET THESE!
loda_master_path = "./Loda"
loda_path = joinpath(loda_master_path, "public/datasets/numerical/")
export_path = "./data" # master path where data will be stored

include("utils.jl")

end