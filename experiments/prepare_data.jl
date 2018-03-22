# either change the settings in the file or call the scrip with arguments
# julia prepare_data.jl dataset alpha difficulty frequency variation seed
# e. g. julia prepare_data.jl iris 0.7 normal 0.02 low 12345

# dependencies
using JLD
push!(LOAD_PATH, "../src")
using AnomalyDetection

# paths
loda_path = "../../../data/Loda/public/datasets/numerical/"
export_path = "./data" # master path where data will be stored

# settings
nargs = size(ARGS, 1)
# dataset name
(nargs > 0)? dataset_name = ARGS[1] : dataset_name = "iris"
# ratio of training to all data
(nargs > 1)? alpha = parse(Float64, ARGS[2]) : alpha = 0.8 
# easy/medium/hard/very_hard problem based on similarity of anomalous measurements to normal
(nargs > 2)? difficulty = ARGS[3] : difficulty = "easy" 
# ratio of anomalous to normal data\n
(nargs > 3)? frequency = parse(Float64, ARGS[4]) : frequency = 0.05 
# low/high - should anomalies be clustered or not
(nargs > 4)? variation = ARGS[5] : variation = "low"
# random seed 
(nargs > 5)? seed = parse(Int64, ARGS[6]) : seed = false 

# load basic dataset and call makeset to extract testing and training data
basicset = Basicset(joinpath(loda_path, dataset_name))
trdata, tstdata, clusterdness = AnomalyDetection.makeset(basicset, alpha, difficulty, frequency, variation,
	seed = seed)

# now save it
fname = joinpath(export_path, string(dataset_name, "_", alpha, "_", difficulty, "_", frequency,
	"_", variation))
mkpath(fname)
save(joinpath(fname, "training_data.jld"), "dataset", trdata)
save(joinpath(fname, "testing_data.jld"), "dataset", tstdata)

println("Data saved to ", fname)