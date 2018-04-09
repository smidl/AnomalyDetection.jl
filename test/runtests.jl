push!(LOAD_PATH, "../src")
using AnomalyDetection
using Base.Test

# run either all tests or just thos specified in command-line call
includes = ["ae", "vae"]
(size(ARGS,1) > 0)? includes = intersect(includes, ARGS) : println("Running all tests...")
@testset "ALL" begin
	for incl in includes
		include(string(incl, ".jl"))
	end
end