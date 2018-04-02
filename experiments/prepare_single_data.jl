# either change the settings in the file or call the script with arguments
# julia prepare_single_data.jl dataset alpha difficulty frequency variation seed
# e. g. julia prepare_single_data.jl iris 0.7 easy 0.02 low 12345

push!(LOAD_PATH, ".") 	
using Experiments

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
# number of repetitions
(nargs > 5)? repetition = parse(Int64, ARGS[6]) : repetition = 0 
# random seed 
(nargs > 6)? seed = parse(Int64, ARGS[7]) : seed = false 

Experiments.prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed,
	repetition = repetition, verb = true)
