# takes a .mat file with stored inputs and settings and produces
# a fit using one of the model from the AnomalyDetection library
# examples of matlab inputs are in the examples dir

# run this in the following way:
# julia matparams.jl /path/to/inputs.mat /path/to/outputs.mat

include("matlab_wrapper.jl")

nargs = length(ARGS)

(nargs>0)? infile = ARGS[1] : error("Specify the input file in the first cl parameter. 
	Run e.g. as julia matparams.jl /path/to/inputs.mat /path/to/outputs.mat")

(nargs>1)? outfile = ARGS[1] : error("Specify the output file in the second cl parameter. 
	Run e.g. as julia matparams.jl /path/to/inputs.mat /path/to/outputs.mat")

auroc, tprvec, fprvec = mat_params(infile, outfile)

