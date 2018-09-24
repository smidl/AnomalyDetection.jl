using Distances
using Lazy
using IterTools
using FileIO
using AnomalyDetection
using DataStructures

@everywhere begin
	algorithms = ["AE", "VAE", "GAN", "fmGAN"]
#	algorithms = ["VAEensemble"]

	if "IsoForest" in algorithms
		println("For Isolation Forest, paralell run is not implemented. Run without the -p flag.")
		isoforest = true
		include("../benchmarks/isolation_forest.jl")
	else
		isoforest = false
	end

	loda_path = "../dataset_analysis/tsne_2D-data"
	host = gethostname()
	#master path where data will be stored
	if host == "vit"
		export_path = "/home/vit/vyzkum/anomaly_detection/data/benchmarks_selected/tsne_2D-experiment/data" 
	elseif host == "axolotl.utia.cas.cz"
		export_path = "home/skvara/work/anomaly_detection/data/benchmarks_selected/tsne_2D-experiment/data"
	end
	include("parallel_utils.jl")
end

iteration = (size(ARGS,1) >0) ? parse(Int64, ARGS[1]) : 1
nhdims = (size(ARGS,1) >0) ? parse(Int64, ARGS[2]) : 1

datasets = ["abalone", "glass", "haberman", "ionosphere", "isolet", "miniboone", 
"multiple-features", "musk-2", "page-blocks", "pendigits"]

pmap(x -> runexperiments(x[1], iteration, x[2], nhdims),
	product(datasets, algorithms))
