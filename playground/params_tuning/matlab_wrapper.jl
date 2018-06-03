# include the library
using AnomalyDetection
using MAT
using EvalCurves

# include auxiliary functions
include("../../experiments/parallel_utils.jl")

# local loda path
loda_path = "../data"

"""
	mat_params(infile, outfile)

The outermost layer. Takes an adress of a .mat file with inputs
and writes output to a .mat file. Also returns the testing auroc,
TP an FP vectors.
"""
function mat_params(infile, outfile)
	# first, load the .mat params
	matparams = matread(infile)

	auroc, tprvec, fprvec = test_params(matparams)

	# save into outfile
	res = Dict("auroc" => auroc, "tprvec" => tprvec, "fprvec" => fprvec)
	matwrite(outfile, res)

	# return for debugging
	return auroc, tprvec, fprvec
end

"""
	test_params(matparams)

The second layer. Takes the inputs specified in a dictionary and 
returns the testing auroc, TP an FP vectors.
"""
function test_params(matparams)
	# get the data
	data = get_data(matparams["dataset"], 12345, true)

	# get the model constructing, anomaly score and fit functions and model
	# parameters
	mf, ff, asf, model_params = functions_params(data, matparams)

	# construct, fit and score
	tras, tstas = construct_fit_score(data, model_params, mf, ff, asf)

	# now compute the final criterion - testing auroc
	tprvec, fprvec = EvalCurves.roccurve(tstas, data[2].labels)
	auroc = EvalCurves.auc(fprvec, tprvec)

	return auroc, tprvec, fprvec
end

"""
	functions_params(data, matparams)

From parameters specified in dictionary, it returns the model constructing,
fitting and anomaly score producing functions and model parameters.
"""
function functions_params(data, matparams)
	# first extract parameters from the large dictionary
	tp = deepcopy(PARAMS[Symbol(matparams["model"])])
	mf = tp[:model]
	ff = tp[:ff]
	asf = tp[:asf]
	model_params_out = tp[:mparams]

	# problem dims
	mpars_in = matparams["model_params"]
	indim, ntr = size(data[1].data[:, data[1].labels.==0])

	# now rewrite the needed values
	# model arch
	lsize, rsize = netsize(mf, indim, mpars_in["ldim"], mpars_in["nhid"])
	set_architecture!(mf, lsize, rsize, model_params_out)

	# batchsize
	(haskey(mpars_in, "batchsize"))? L = mpars_in["batchsize"] : L = 256
	model_params_out[:args][:L] = min(L, ntr)

	# now the rest of the params
	for key in keys(mpars_in)
		if (key == "alpha" && mf == AnomalyDetection.fmGANmodel)
			model_params_out[:kwargs][Symbol(key)] = mpars_in[key]
		elseif !(key in ["nhid", "ldim"])
			model_params_out[:args][Symbol(key)] = mpars_in[key]
		end
	end

	return mf, ff, asf, model_params_out
end

"""
	set_architecture!(mf, lsize, rsize, model_params)

Sets the correct field (gsize, dsize, esize) in the constructor params.
"""
function set_architecture!(mf, lsize, rsize, model_params)
	if mf in [AnomalyDetection.AEmodel, AnomalyDetection.VAEmodel]
		model_params[:args][:esize] = lsize
		model_params[:args][:dsize] = rsize
	elseif mf in [AnomalyDetection.GANmodel, AnomalyDetection.fmGANmodel]
		model_params[:args][:gsize] = lsize
		model_params[:args][:dsize] = rsize
	end
end

"""
	construct_fit_score(data, params, mf, ff, asf, [fitnonly])

Cosntructs the model and calls fit and score.
"""
function construct_fit_score(data, params, mf, ff, asf, fitnonly = true)
	# construct the model
	model = mf([p for p in values(params[:args])]...;
			[p[1] => eval(p[2]) for p in params[:kwargs]]...)

	# fit and score
	return fit_score(data, model, ff, asf, fitnonly)
end

"""
	fit_score(data, model, ff, asf, [fitnonly])

Fits the model and returns anomaly scores.
"""
function fit_score(data, model, ff, asf, fitnonly = true)
	# extract data
	(trdata, tstdata) = data

	# fit
	if fitnonly # fit only on normal data
		ff(model, trdata.data[:,trdata.labels .== 0])
	else
		ff(model, trdata.data)
	end

	# get anomaly scores
	return getas(data, model, asf)
end