using Distances
using Lazy
using IterTools
using FileIO
using AnomalyDetection
using DataStructures

@everywhere begin
	algorithms = ["VAE", "GAN", "fmGAN"]
	#algorithms = ["VAEensemble"]

	if "IsoForest" in algorithms
		println("For Isolation Forest, paralell run is not implemented. Run without the -p flag.")
		isoforest = true
		include("../benchmarks/isolation_forest.jl")
	else
		isoforest = false
	end

	export_path = ARGS[1]
	loda_path = ARGS[2]
	include("parallel_utils.jl")
end

iteration = (size(ARGS,1) >2) ? parse(Int64, ARGS[3]) : 1
nhdims = (size(ARGS,1) >2) ? parse(Int64, ARGS[4]) : 1

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(!(s -> s in ["url", "gisette", "persistent-connection"]))

datasets = datasets[1:1]
map(x -> runexperiments(x[1], iteration, x[2], nhdims),
	product(datasets, algorithms))
