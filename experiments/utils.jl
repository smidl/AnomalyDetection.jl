"""
	prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed, [repetition, verb])

Prepare a single experiment.
"""
function prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed; 
	repetition = 0, verb = false)
	for i in 1:max(repetition,1)
		# load basic dataset and call makeset to extract testing and training data
		basicset = Basicset(joinpath(loda_path, dataset_name))
		trdata, tstdata, clusterdness = AnomalyDetection.makeset(basicset, alpha, difficulty, frequency, variation,
			seed = seed)

		# now save it
		fname = joinpath(export_path, string(lpad(size(trdata.data,1), 4, 0), "_", dataset_name, 
			"_", alpha, "_", difficulty, "_", frequency, "_", variation))
		# if more samples are requested, create subfolders
		if repetition > 0
			fname = joinpath(fname, "$i")
		end
		mkpath(fname)
		save(joinpath(fname, "training_data.jld"), "dataset", trdata)
		save(joinpath(fname, "testing_data.jld"), "dataset", tstdata)

		if verb
			println("Data saved to ", fname)
		end
	end
end

"""
	prepare_experiment_data(repetition)

Set up data for the Loda experiment.
"""
function prepare_experiment_data(repetition)
	# settings
	# ratio of training to all data
	alpha = 0.8 
	# easy/medium/hard/very_hard problem based on similarity of anomalous measurements to normal
	difficulty = "easy" 
	# ratio of anomalous to normal data
	frequency = 0.02 
	# low/high - should anomalies be clustered or not
	variation = "low"
	# random seed 
	seed = false 
	# verbosity of the saving routine
	verb = false

	files = readdir(Experiments.loda_path)

	# export the datasets
	for dataset_name in files
		try
			Experiments.prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed,
				repetition = repetition, verb = verb)
		end
	end

	# now export those datasets that dont have easy difficulty anomalies
	# you can check them using "julia print_loda_overview.jl"
	difficulty = "medium"

	for dataset_name in ["madelon", "gisette"]
		Experiments.prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed,
				repetition = repetition, verb = verb)
	end

	difficulty = "hard"
	for dataset_name in ["vertebral-column"]
		Experiments.prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed,
				repetition = repetition, verb = verb)
	end
end