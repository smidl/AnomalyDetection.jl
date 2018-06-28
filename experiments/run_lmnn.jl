using Distances
using Lazy
using IterTools
using FileIO
using DataStructures
using PyCall, Flux, AnomalyDetection, JLD
@pyimport pylmnn.lmnn as lmnn

@everywhere begin
	loda_path = "./datasets"
	export_path = "./lmnn" #master path where data will be stored
	include("parallel_utils.jl")
	include("lmnn_utils.jl")
end

iteration = (size(ARGS,1) >0) ? parse(Int64, ARGS[1]) : 1

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(!(s -> s in ["url", "gisette", "persistant-connection", 
	"miniboone", "statlog-shuttle"]))

pmap(x -> runexperiment(x[1], iteration, x[2]),
	product(datasets, ["kNN"]))
