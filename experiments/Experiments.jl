module Experiments

using JLD
push!(LOAD_PATH, "../src")
using AnomalyDetection


# paths
# SET THESE!
loda_path = "../../../data/Loda/public/datasets/numerical/"
export_path = "./data" # master path where data will be stored

include("utils.jl")

end