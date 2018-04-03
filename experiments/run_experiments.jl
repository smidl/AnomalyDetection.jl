# runs all the experiments!
push!(LOAD_PATH, ".")
using Experiments

# first export all the data
(size(ARGS,1) >0)? repetition = parse(Int64, ARGS[1]) : repetition = 1
Experiments.prepare_experiment_data(repetition)

# extract all created data folders
fpaths = joinpath.(Experiments.export_path, readdir(Experiments.export_path))

# train all the algorithms
# go folder by folder and in each one train all the algorithms and compute anomaly scores
println("precompiling...")
Experiments.run_experiment(fpaths[1:1], "compile")
println("done\n")
println("Running the experiment.")
@time Experiments.run_experiment(fpaths, "run")