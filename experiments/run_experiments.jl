# runs all the experiments!
push!(LOAD_PATH, ".")
using Experiments

# first export all the data
repetition = 1
Experiments.prepare_experiment_data(repetition)
