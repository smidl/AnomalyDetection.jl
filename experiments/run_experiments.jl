# runs all the experiments!
# run with a parameter stating number of samples taken from Loda data, e.g.
# julia run_experiments.jl 2
# which will be taken from all the datasets and all the algorithms will be run in these separate
# folders
push!(LOAD_PATH, ".")
using Experiments

# download loda datasets
Experiments.downloadloda()

# first export all the data
(size(ARGS,1) >0)? repetition = parse(Int64, ARGS[1]) : repetition = 1
#Experiments.prepare_experiment_data(repetition)

# extract all created data folders
fpaths = joinpath.(Experiments.export_path, readdir(Experiments.export_path))

# train all the algorithms
# go folder by folder and in each one train all the algorithms and compute anomaly scores
println("precompiling...")
#Experiments.run_experiment(fpaths[1:1], "compile")
println("done\n")
println("Running the experiment.")
#@time Experiments.run_experiment(fpaths, "run")