using Distances
using Lazy
using Iterators
using FileIO
# this shoul not be necessary when the AnomalyDetection repo is renamed
script_abs_path = replace(@__FILE__, split(@__FILE__, "/")[end], "")
push!(LOAD_PATH, joinpath(script_abs_path, "../src"))
using AnomalyDetection

@everywhere begin
	loda_path = "/mnt/output/data/datasets/numerical"
	export_path = "/mnt/output/anomaly" #master path where data will be stored
	include(joinpath(script_abs_path, "parallel_utils.jl"))
end

dpaths = joinpath.(export_path, readdir(export_path))
dpaths = @>> dpaths mapreduce(s -> joinpath.(s,readdir(s)),vcat) filter(isdir)
pmap(i -> i[1](i[2],i[3]),product([trainAE, trainVAE, trainsVAE, trainGAN, trainfmGAN, trainkNN],dpaths,1))