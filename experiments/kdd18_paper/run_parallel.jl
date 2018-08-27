# version for experiments run on AWS
using Distances
using Lazy
using IterTools
using FileIO
using AnomalyDetection
using DataStructures

@everywhere begin
	algorithms = ["kNN" ,"AE", "GAN", "fmGAN", "VAE"]

	if "IsoForest" in algorithms
		println("For Isolation Forest, paralell run is not implemented. Run without the -p flag.")
		isoforest = true
		include("isolation_forest.jl")
	else
		isoforest = false
	end

	loda_path = "/opt/output/data/datasets/numerical"
	export_path = "/opt/output/anomaly" #master path where data will be stored
	include("/opt/src/AnomalyDetection/experiments/parallel_utils.jl")
end

iteration = (size(ARGS,1) >0) ? parse(Int64, ARGS[1]) : 1
nhdims = (size(ARGS,1) >0) ? parse(Int64, ARGS[2]) : 1

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(!(s -> s in ["url", "gisette", "persistant-connection"]))

pmap(x -> runexperiments(x[1], iteration, x[2], nhdims),
	product(datasets, algorithms))
