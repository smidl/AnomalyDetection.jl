using PyPlot
using JLD
using DataFrames
using CSV
using AnomalyDetection

# this is where the LODA datasets are stored, change if necessary
data_path = joinpath(@__DIR__, "datasets")

function overview()
	# data
	datasets = AnomalyDetection.loaddata(data_path);

	# fill the dataset
	arr = Array{Any, 2}(undef, length(datasets), 7)
	i = 1
	for key in keys(datasets)
	    arr[i, 1] = key
	    dataset = datasets[key]
	    arr[i,2] = size(dataset.normal,1)
	    arr[i,3] = size(dataset.normal,2)
	    arr[i,4] = size(dataset.easy,2)
	    arr[i,5] = size(dataset.medium,2)
	    arr[i,6] = size(dataset.hard,2)
	    arr[i,7] = size(dataset.very_hard,2)
	#    println("$key: M = $(size(dataset.normal,1)), #n = $(size(dataset.normal,2)), ",
	#        "#e = $(size(dataset.easy,2)), #m = $(size(dataset.medium,2)), ",
	#        "#h = $(size(dataset.hard,2)), #vh = $(size(dataset.very_hard,2))")
	    i+=1
	end
	df = convert(DataFrame, arr)
	rename!(df, f => t for (f, t) = 
	    zip([:x1, :x2, :x3, :x4, :x5, :x6, :x7], 
	        [:name, :M, :normal, :easy, :medium, :hard, :very_hard]))
	return sort(df, :M)
end
df = overview()
print(df)
println("")
CSV.write("loda_overview.csv", df)