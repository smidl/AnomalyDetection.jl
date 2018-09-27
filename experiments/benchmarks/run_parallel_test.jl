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

	export_path = "./test"
	loda_path = "../dataset_analysis/tsne_2D-data"
	include("parallel_utils.jl")
end

iteration = 1
nhdims = 1

datasets = @>> readdir(loda_path) filter(s -> isdir(joinpath(loda_path,s))) filter(!(s -> s in ["url", "gisette", "persistent-connection"]))

datasets = datasets[1:1]
map(x -> runexperiments(x[1], iteration, x[2], nhdims),
	product(datasets, algorithms))
