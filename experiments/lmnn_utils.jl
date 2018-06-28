"""
	dataset_lmnn(dataset)

For a dataset, compute the lmnn and return the tranformation matrix L and the transformed data.
"""
function dataset_lmnn(data, k)
	(trdata, tstdata) = data
	X = cat(2,trdata.data,tstdata.data)'
	y = cat(1, trdata.labels, tstdata.labels)

	N,M = size(X)

	max_iter = 200
	dim_out = M

	clf = lmnn.LargeMarginNearestNeighbor(n_neighbors = k, max_iter=max_iter, 
			n_features_out=dim_out, verbose = 0)	
	clf[:fit](X,y)
	L = copy(clf[:L_])
	
	trLdata = AnomalyDetection.Dataset(L*trdata.data, trdata.labels)
	tstLdata = AnomalyDetection.Dataset(L*tstdata.data, tstdata.labels)
	return (trLdata, tstLdata), L
end

"""
	experiment(data, mf, mfp, ff, ffp, asf, asfp, outpath, fname)

Runs an experiment.
data = tuple of (training data, testing data)
mf = model constructor (function)
mfp = model constructor parameter, contaning :args (SortedDictionary) and
	:kwargs (Dictionary), also this parameter structure will be updated in each
	iteration and saved to output
ff = fit function of syntax fit = ff(model, X)
ffp = an iterable of fit parameters with which the model will be updated in the outer
	loop, e.g.

	product([:batchsize => i for i in [100, 200]], [:lambda => i for i in [1e-2, 1e-3]])

	then the model is updated "model.batchsize = 100; model.lambda = 1e-2 ..."
asf = anomaly score function of syntax anomalyscore = asf(model, X)
asfp = an iterable of as parameters with which the model will be updated in the inner
	loop, e.g.

	product([:alpha => i for i in [0.1, 0.9]], [:M => i for i in 1:10])

	then the model is updated "model.alpha = 0.1; model.M = 1 ..."
"""
function experiment(data, mf, mfp, ff, ffp, asf, asfp, outpath, fname)
	# outer loop over fit parameters
	for fargs in ffp
		# mparams is the dictionary of used parameters that will be saved
		mparams = deepcopy(mfp) # so that the original params are not owerwritten
		# universal model constructor
		model = mf([p for p in values(mparams[:args])]...;
			[p[1] => eval(p[2]) for p in mparams[:kwargs]]...)
		# SortedDict causes problems during load(), now the order does not matter anymore
		mparams[:args] = Dict(mparams[:args])

		# update model parameters, filename and save actual values to mparams
		_fname = updateparams!(model, fname, fargs, mparams)

		# extract and convert datasets

		# inner loop over anomaly score parameters
		for args in asfp
			(trdata, tstdata), _ = dataset_lmnn(data, args[1][2])
			println(args[1][2]) 
			ndata = trdata.data[:,trdata.labels .== 0] # normal training data

			# fit the model on normal data
			_,ft,_,_,_ = @timed ff(model, ndata)

			# update model parameters, filename and save actual values to mparams
			__fname = updateparams!(model, _fname, args, mparams)

			# get anomaly scores
			(tras, tstas), pt, _, _, _ = @timed getas(data, model, asf)

			# save input and output
			__fname = string(__fname, ".jld")
			save_io(outpath, __fname, model, mparams, tras, trdata.labels, tstas,
				tstdata.labels, ft, pt)
		end
	end
end
