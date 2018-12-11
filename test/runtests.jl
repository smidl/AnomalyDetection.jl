using AnomalyDetection, Flux, ValueHistories
using Test, Random

# common test inputs
# input data setup
xdim = 2
latentdim = 4
hiddendim = 8
N = 10
# set seed and create normal data with one anomaly
Random.seed!(123)
X = randn(xdim, N)
X[:,end] .+= 10.0
X = AnomalyDetection.Float.(X)
nX = X[:,1:end-1]
Y = Int.(push!(zeros(N-1), 1))

# run either all tests or just those specified in command-line call
includes = ["ae", "vae"]
(size(ARGS,1) > 0) ? includes = intersect(includes, ARGS) : println("Running all tests...")
@testset "ALL" begin
	for incl in includes
		include(string(incl, ".jl"))
	end
end