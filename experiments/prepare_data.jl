# contains the basic prepare_data() function
# dependencies
using JLD
push!(LOAD_PATH, "../src")
using AnomalyDetection

# paths
loda_path = "../../../data/Loda/public/datasets/numerical/"
export_path = "./data" # master path where data will be stored

function prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed)
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
end
