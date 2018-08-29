###########################
### general NN settings ###
###### run settings #######
const hiddendim = 16
const latentdim = 8
const activation = Flux.relu
const verbfit = false
const batchsizes = [256]
const useepochs = true
###############################

"""
	save_io(path, params, ascore, labels, algo_name)

Saves an algorithm output and input - params and anomaly scores.
"""
function save_io(path, file, mparams, trascore, trlabels, tstascore, tstlabels, loss,
	alg_label, nnparams, fittime, predicttime)
	mkpath(path)
	FileIO.save(joinpath(path, file), "params", mparams, "training_anomaly_score", trascore,
		"training_labels", trlabels, "testing_anomaly_score", tstascore,
		"testing_labels", tstlabels, "loss", loss, "algorithm", alg_label,
		"NN_params", nnparams, "fit_time", fittime, "predict_time", predicttime)
end

# model-specific saving routines
save_io(path, file, m, mparams, tras, trl, tstas, tstl, ft, pt) =
	save_io(path, file, mparams, tras, trl, tstas, tstl, nothing, string(m), nothing, ft, pt)
save_io(path, file, m::AnomalyDetection.kNN, mparams, tras, trl, tstas, tstl, ft, pt) =
	save_io(path, file, mparams, tras, trl, tstas, tstl, nothing, string(m), nothing, ft, pt)
save_io{model<:AnomalyDetection.genmodel}(path, file, m::model, mparams, tras, trl, tstas,
	tstl, ft, pt) =
	save_io(path, file, mparams, tras, trl, tstas, tstl, nothing, string(m),
		nothing, ft, pt)

"""
	get_data(dataset_name, iteration, allanomalies)

Returns training and testing dataset.
"""
function get_data(dataset_name, iteration, allanomalies=false)
	# get the dataset
	basicset = AnomalyDetection.Basicset(joinpath(loda_path, dataset_name))
	# settings

	# ratio of training to all data
	alpha = 0.8

	# random seed
	seed = Int64(iteration)
	
	if !allanomalies
		# easy/medium/hard/very_hard problem based on similarity of anomalous measurements to normal
		# some datasets dont have easy difficulty anomalies
		if dataset_name in ["madelon", "gisette", "abalone", "haberman", "letter-recognition",
			"isolet", "multiple-features", "statlog-shuttle"]
			difficulty = "medium"
		elseif dataset_name in ["vertebral-column"]
			difficulty = "hard"
		else
			difficulty = "easy"
		end
		# ratio of anomalous to normal data
		frequency = 0.05
		# low/high - should anomalies be clustered or not
		variation = "low"

		# this might fail for url
		trdata, tstdata, clusterdness = AnomalyDetection.makeset(basicset, alpha, difficulty,
			frequency, variation, seed = seed)
	else
		trdata, tstdata, clusterdness = AnomalyDetection.makeset(basicset, alpha, seed = seed)
	end

	return trdata, tstdata
end

"""
	getas(dataset_name, m, as)

Produces anomaly scores on testing dataset.
m = model
as = anomaly score function
"""
function getas(data, m, as)
	# get the data
	(trdata, tstdata) = data

	# now loop over all ascoreargs
	tras = as(m, trdata.data)
	tstas = as(m, tstdata.data)
	return tras, tstas
end

"""
	runexperiment(dataset_name, iteration, alg)

Extracts model parameters and iterables from global const PARAMS,
creates, trains and predicts anomaly scores for model alg for each
parameter setting.
"""
function runexperiment(dataset_name, iteration, alg)
	# load data
	data = get_data(dataset_name, iteration)

	# top level of the param tree
	tp = deepcopy(PARAMS[Symbol(alg)])

	# data will be saved here
	path = 	joinpath(export_path, string("$(dataset_name)/$(alg)/$(iteration)"))

	# upgrade params according to data size
	dataparams!(tp[:model], tp, data)

	# run the experiment
	experiment(data, tp[:model], tp[:mparams], tp[:ff], tp[:ffparams],
		tp[:asf], tp[:asfparams], path, alg)

	println("Training of $alg on $path finished!")
end

"""
	runexperiments(dataset_name, iteration, alg)

Extracts model parameters and iterables from global const PARAMS,
creates, trains and predicts anomaly scores for model alg for each
parameter setting. Multiple architecture version.
"""
function runexperiments(dataset_name, iteration, alg, nhid)
	# load data
	data = get_data(dataset_name, iteration)

	# top level of the param tree
	tp = deepcopy(PARAMS[Symbol(alg)])

	# data will be saved here
	path = 	joinpath(export_path, string("$(dataset_name)/$(alg)/$(iteration)"))

	latentdims, indim = getlatentdims(data)

	# loo over different
	for latentdim in latentdims
		fbase = updatearchitecture!(tp[:model], tp, indim, latentdim, nhid)

		# upgrade params according to data size
		dataparams!(tp[:model], tp, data)

		# run the experiment
		experiment(data, tp[:model], tp[:mparams], tp[:ff], tp[:ffparams],
			tp[:asf], tp[:asfparams], path, string(alg, "_", fbase))

	end
	println("Training of $alg on $path finished!")
end

function getlatentdims(data)
	indim, trN = size(data[1].data[:,data[1].labels.==0])
	maxd = min(200, indim)
	latentdims = unique(Int.(ceil.([i/10 for i in 1:5]*maxd)))
	return latentdims, indim
end

function netsize(m::Type{AnomalyDetection.AEmodel}, indim, ldim, nhid)
	# create linearly spaced layer sizes
	dsize = [i for i in Int.(round.(linspace(ldim, indim, nhid+3)))]
	esize = dsize[nhid+3:-1:1]
	return esize, dsize
end

function updatearchitecture!(m::Type{AnomalyDetection.AEmodel}, tp,
	indim, ldim, nhid)
	esize, dsize = netsize(m, indim, ldim, nhid)
	tp[:mparams][:args][:dsize] = dsize
	tp[:mparams][:args][:esize] = esize
	# create the name string
	f = "esize"
	for s in esize
		f=string(f,"-$(s)")
	end
	f=string(f,"_dsize")
	for s in dsize
		f=string(f,"-$(s)")
	end
	return f
end

function netsize(m::Type{AnomalyDetection.VAEmodel}, indim, ldim, nhid)
	# create linearly spaced layer sizes
	dsize = [i for i in Int.(round.(linspace(ldim, indim, nhid+3)))]
	esize = dsize[nhid+3:-1:1]
	esize[end] = 2*esize[end]
	return esize, dsize
end

function updatearchitecture!(m::Type{AnomalyDetection.VAEmodel}, tp,
	indim, ldim, nhid)
	esize, dsize = netsize(m, indim, ldim, nhid)

	tp[:mparams][:args][:dsize] = dsize
	tp[:mparams][:args][:esize] = esize
	# create the name string
	f = "esize"
	for s in esize
		f=string(f,"-$(s)")
	end
	f=string(f,"_dsize")
	for s in dsize
		f=string(f,"-$(s)")
	end
	return f
end

function netsize(m::Type{AnomalyDetection.sVAEmodel}, indim, ldim, nhid)
	# create linearly spaced layer sizes
	decsize = [i for i in Int.(round.(linspace(ldim, indim, nhid+3)))]
	ensize = decsize[nhid+3:-1:1]
	ensize[end] = 2*ensize[end]
	dissize = [i for i in Int.(round.(linspace(indim+latentdim, 1, nhid+3)))]
	return decsize, ensize, dissize
end

function updatearchitecture!(m::Type{AnomalyDetection.sVAEmodel}, tp,
	indim, ldim, nhid)
	decsize, ensize, dissize = netsize(m, indim, ldim, nhid)

	tp[:mparams][:args][:decsize] = decsize
	tp[:mparams][:args][:ensize] = ensize
	tp[:mparams][:args][:dissize] = dissize
	# create the name string
	f = "ensize"
	for s in ensize
		f=string(f,"-$(s)")
	end
	f=string(f,"_decsize")
	for s in decsize
		f=string(f,"-$(s)")
	end
	f=string(f,"_dissize")
	for s in dissize
		f=string(f,"-$(s)")
	end
	return f
end

function netsize(m::Union{Type{AnomalyDetection.GANmodel},
	Type{AnomalyDetection.fmGANmodel}}, indim, ldim, nhid)
	# create linearly spaced layer sizes
	gsize = [i for i in Int.(round.(linspace(ldim, indim, nhid+3)))]
	dsize = gsize[nhid+3:-1:1]
	dsize[end] = 1
	return gsize, dsize
end

function updatearchitecture!(m::Union{Type{AnomalyDetection.GANmodel},
	Type{AnomalyDetection.fmGANmodel}}, tp,
	indim, ldim, nhid)
	gsize, dsize = netsize(m, indim, ldim, nhid)

	tp[:mparams][:args][:gsize] = gsize
	tp[:mparams][:args][:dsize] = dsize
	# create the name string
	f = "gsize"
	for s in gsize
		f=string(f,"-$(s)")
	end
	f=string(f,"_dsize")
	for s in dsize
		f=string(f,"-$(s)")
	end
	return f
end

function updatearchitecture!(m, tp,
	indim, ldim, nhid)
	return ""
end

"""
	experiment(data, mf, mfp, ff, ffp, asf, asfp, outpath, fname)

Runs an experiment.\n
data = tuple of (training data, testing data)\n
mf = model constructor (function)\n
mfp = model constructor parameter, contaning :args (SortedDictionary) and\n
	:kwargs (Dictionary), also this parameter structure will be updated in each\n
	iteration and saved to output\n
ff = fit function of syntax fit = ff(model, X)\n
ffp = an iterable of fit parameters with which the model will be updated in the outer\n
	loop, e.g.\n
\n
	product([:batchsize => i for i in [100, 200]], [:lambda => i for i in [1e-2, 1e-3]])\n
\n
	then the model is updated "model.batchsize = 100; model.lambda = 1e-2 ..."\n
asf = anomaly score function of syntax anomalyscore = asf(model, X)\n
asfp = an iterable of as parameters with which the model will be updated in the inner\n
	loop, e.g.\n
\n
	product([:alpha => i for i in [0.1, 0.9]], [:M => i for i in 1:10])\n
\n
	then the model is updated "model.alpha = 0.1; model.M = 1 ..."\n
"""
function experiment(data, mf, mfp, ff, ffp, asf, asfp, outpath, fname)
	# extract datasets
	(trdata, tstdata) = data
	ndata = trdata.data[:,trdata.labels .== 0] # normal training data

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

		# fit the model on normal data
		_,ft,_,_,_ = @timed ff(model, ndata)

		# inner loop over anomaly score parameters
		for args in asfp
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

"""
	updateparams!(model, fname, args, mparams)

Updates params of the model, values of params dictionary to be saved and the filename
according to args iterable.	Outputs the modified filename.
"""
function updateparams!(model, fname, args, mparams)
	for a in args
		setfield!(model, a...)
		fname = string(fname, "_$(a[1])-$(a[2])")
		mparams[:kwargs][a[1]] = a[2]
	end
	return fname
end

"""
	setnepochs(topparams, N)

Set the number of epochs so that there is still a set number of iterations going on.
"""
function setnepochs(topparams, N)
	if useepochs
		iters = topparams[:mparams][:kwargs][:iterations]
		batchsize = batchsizes[1]
		topparams[:mparams][:kwargs][:nepochs] = 
			Int(ceil(batchsize*iters/N))
	else
		topparams[:mparams][:kwargs][:nepochs] = nothing
	end
end

"""
	dataparams!(model, topparams, data)

Define operations needed to be done with params according to data prior to
training (e.g. batchsize, input dimensions). Defaults to doing nothing.
"""
dataparams!(model, topparams, data) = return nothing

function dataparams!(model::Type{AnomalyDetection.kNN}, topparams, data)
	# for kNN model, set the ks according to data size
	indim, trN = size(data[1].data[:,data[1].labels.==0])
	topparams[:asfparams] = IterTools.product([:k => i for i in Int.(round.(linspace(1, 2*sqrt(trN), 5)))])
end

function dataparams!{model<:AnomalyDetection.genmodel}(m::Type{model}, topparams, data)
	# change the esize and dsize params based on data size
	indim, trN = size(data[1].data[:,data[1].labels.==0])
	topparams[:mparams][:args][:esize][1] = indim
	topparams[:mparams][:args][:dsize][end] = indim

	# set epoch number if needed
	setnepochs(topparams, trN)

	# modify the batchsizes
	databatchsize!(trN, topparams)
end

function dataparams!(m::Type{AnomalyDetection.VAEmodel}, topparams, data)
	# change the esize and dsize params based on data size
	indim, trN = size(data[1].data[:,data[1].labels.==0])
	topparams[:mparams][:args][:esize][1] = indim
	if eval(topparams[:mparams][:kwargs][:variant]) == :unit
		topparams[:mparams][:args][:dsize][end] = indim
	else
		topparams[:mparams][:args][:dsize][end] = indim*2
	end

	# set epoch number if needed
	setnepochs(topparams, trN)

	# modify the batchsizes
	databatchsize!(trN, topparams)
end

function dataparams!(m::Type{AnomalyDetection.sVAEmodel}, topparams, data)
	# change the esize and dsize params based on data size
	indim, trN = size(data[1].data[:,data[1].labels.==0])
	topparams[:mparams][:args][:ensize][1] = indim
	topparams[:mparams][:args][:decsize][end] = indim
	# indim + latentdim
	topparams[:mparams][:args][:dissize][1] = indim +
		topparams[:mparams][:args][:decsize][1]

	# set epoch number if needed
	setnepochs(topparams, trN)
	
	# modify the batchsizes
	databatchsize!(trN, topparams)
end

function dataparams!(m::Type{AnomalyDetection.GANmodel}, topparams, data)
	# change the esize and dsize params based on data size
	indim, trN = size(data[1].data[:,data[1].labels.==0])
	topparams[:mparams][:args][:gsize][end] = indim
	topparams[:mparams][:args][:dsize][1] = indim

	# set epoch number if needed
	setnepochs(topparams, trN)
	
	# modify the batchsizes
	databatchsize!(trN, topparams)
end

function dataparams!(m::Type{AnomalyDetection.fmGANmodel}, topparams, data)
	# change the esize and dsize params based on data size
	indim, trN = size(data[1].data[:,data[1].labels.==0])
	topparams[:mparams][:args][:gsize][end] = indim
	topparams[:mparams][:args][:dsize][1] = indim

	# set epoch number if needed
	setnepochs(topparams, trN)
	
	# modify the batchsizes
	databatchsize!(trN, topparams)
end

"""
	databatchsize!(N, topparams)

Modify batchsize L according to data size.
N - number of instances
topparams - top of parameter tree
"""
function databatchsize!(N, topparams)
	# modify the batchsize if it is larger than the dataset size
	# this is a little awkward but universal
	# steps:
	# 1) if there is an L larger than datasize N, create a new pair :L => trN
	# 2) filter out those pairs where :L > trN
	# 3) remove duplicates (if there are more pairs with :L > trN)
	ls = Array{Any,1}([x for x in topparams[:ffparams].xss])
	for l in ls
		map(x -> ((x[1]==:batchsize && x[2] > N)? push!(l, (x[1] => N)) : (nothing)), l)
	end

	for i in 1:size(ls,1)
	 	ls[i] = filter(x -> !(x[1]==:batchsize && x[2] > N), ls[i])
	    ls[i] = unique(ls[i])
	end
	topparams[:ffparams] = IterTools.product(ls...)
end

_unit() = :unit
_sigma() = :sigma

PARAMS = Dict(
	### kNN ###
	:kNN => Dict(
		# this serves as model construction params and also to be saved
		# in io
		:mparams => Dict(
			# args for the model constructor, must be in correct order
			:args => DataStructures.OrderedDict(
				:k => 1, # will be replaced from the asfparams
				:contamination => 0.0 # useless
				),
			# kwargs
			:kwargs => Dict(
				:metric => :(Distances.Euclidean()),
				:distances => "all",
				:threshold => 0.0, # useless
				:reduced_dim => false
				)
			),
		# this is going to be iterated ver for the fit function
		:ffparams => IterTools.product(),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => IterTools.product([:k => i for i in [1, 3, 5, 11, 21]]),
		# the model constructor
		:model => AnomalyDetection.kNN,
		# model fit function
		:ff => AnomalyDetection.fit!,
		# anomaly score function
		:asf => AnomalyDetection.anomalyscore
		),	
	### AE ###
	:AE => Dict(
	# this is going to serve as model construction params and also to be saved
	# in io
		:mparams => Dict(
			# args for the model constructor, must be in correct order
			:args => DataStructures.OrderedDict(
				:esize => [1; hiddendim; hiddendim; latentdim],
				:dsize => [latentdim; hiddendim; hiddendim; 1]
				),
			# kwargs
			:kwargs => Dict(
				:batchsize => 0, # replaced in training
				:threshold => 0, # useless
				:contamination => 0, # useless
				:iterations => 10000,
				:cbit => 1000,
				:verbfit => verbfit,
				:nepochs => nothing,
				# input functions like this, they are evaluated later (only here)
				:activation => :(Flux.relu),
				:layer => :(Flux.Dense),
				:tracked => true,
				:rdelta => 1e-5,
				:eta => 0.001
				)
			),
		# this is going to be iterated over for the fit function
		:ffparams => IterTools.product([:batchsize => i for i in batchsizes]),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => IterTools.product(),
		# the model constructor
		:model => AnomalyDetection.AEmodel,
		# model fit function
		:ff => AnomalyDetection.fit!,
		# anomaly score function
		:asf => AnomalyDetection.anomalyscore
		),
	### VAE ###
	:VAE => Dict(
	# this is going to serve as model construction params and also to be saved
	# in io
		:mparams => Dict(
			# args for the model constructor, must be in correct order
			:args => DataStructures.OrderedDict(
				:esize => [1; hiddendim; hiddendim; latentdim*2],
				:dsize => [latentdim; hiddendim; hiddendim; 1]
				),
			# kwargs
			:kwargs => Dict(
				:lambda => 1, # replaced in training
				:threshold => 0, # useless
				:contamination => 0, # useless
				:iterations => 10000,
				:cbit => 1000,
				:verbfit => verbfit,
				:batchsize => 0, # replaced in training
				:M => 1,
				# input functions like this, they are evaluated later
				:activation => :(Flux.relu),
				:layer => :(Flux.Dense),
				:tracked => true,
				:rdelta => Inf,
				:astype => "likelihood",
				:eta => 0.001,
				:variant => :(_unit())
				)
			),
		# this is going to be iterated over for the fit function
		:ffparams => IterTools.product([:batchsize => i for i in batchsizes],
							[:lambda => i for i in [10.0^i for i in 0:-1:-4]]),
							#[:lambda => i for i in [10.0^i for i in 0:0]]),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => IterTools.product([:astype => s for s in ["likelihood"]]),
		# the model constructor
		:model => AnomalyDetection.VAEmodel,
		# model fit function
		:ff => AnomalyDetection.fit!,
		# anomaly score function
		:asf => AnomalyDetection.anomalyscore
		),
	### sVAE ###
	:sVAE => Dict(
	# this is going to serve as model construction params and also to be saved
	# in io
		:mparams => Dict(
			# args for the model constructor, must be in correct order
			:args => DataStructures.OrderedDict(
				:ensize => [1; hiddendim; hiddendim; latentdim*2],
				:decsize => [latentdim; hiddendim; hiddendim; 1],
				:dissize => [1 + latentdim; hiddendim; hiddendim; 1]
				),
			# kwargs
			:kwargs => Dict(
				:lambda => 0, # replaced in training
				:threshold => 0, # useless
				:contamination => 0, # useless
				:iterations => 10000,
				:cbit => 1000,
				:nepochs => nothing,
				:verbfit => verbfit,
				:batchsize => 0, # replaced in training
				:M => 1,
				# input functions like this, they are evaluated later
				:activation => :(Flux.relu),
				:layer => :(Flux.Dense),
				:tracked => true,
				:rdelta => 1e-5,
				:alpha => 0, # will be replaced in training
				:xsigma => 1.0
				)
			),
		# this is going to be iterated over for the fit function
		:ffparams => IterTools.product([:batchsize => i for i in batchsizes],
							 [:lambda => i for i in [0.0; [10.0^i for i in -4:2:4]]]),
# 							 [:lambda => i for i in [0.0]]),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => IterTools.product([:alpha => i for i in linspace(0,1,6)]),
		# the model constructor
		:model => AnomalyDetection.sVAEmodel,
		# model fit function
		:ff => AnomalyDetection.fit!,
		# anomaly score function
		:asf => AnomalyDetection.anomalyscore
		),
	### GAN ###
	:GAN => Dict(
	# this is going to serve as model construction params and also to be saved
	# in io
		:mparams => Dict(
			# args for the model constructor, must be in correct order
			:args => DataStructures.OrderedDict(
				:gsize => [latentdim; hiddendim; hiddendim; 1],
				:dsize => [1; hiddendim; hiddendim; 1],
				),
			# kwargs
			:kwargs => Dict(
				:lambda => 0.5, # replaced in training
				:threshold => 0, # useless
				:contamination => 0, # useless
				:batchsize => 0, # replaced in training
				:iterations => 10000,
				:cbit => 1000,
				:nepochs =>nothing,
				:verbfit => verbfit,
				# input functions like this, they are evaluated later
				:activation => :(Flux.relu),
				:layer => :(Flux.Dense),
				:tracked => true,
				:rdelta => 1e-5,
				:pz => :(randn),
				:eta => 0.001
				)
			),
		# this is going to be iterated over for the fit function
		:ffparams => IterTools.product([:batchsize => i for i in batchsizes]),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => IterTools.product([:lambda => i for i in linspace(0,1,6)]),
		# the model constructor
		:model => AnomalyDetection.GANmodel,
		# model fit function
		:ff => AnomalyDetection.fit!,
		# anomaly score function
		:asf => AnomalyDetection.anomalyscore
		),
	### fmGAN ###
	:fmGAN => Dict(
	# this is going to serve as model construction params and also to be saved
	# in io
		:mparams => Dict(
			# args for the model constructor, must be in correct order
			:args => DataStructures.OrderedDict(
				:gsize => [latentdim; hiddendim; hiddendim; 1],
				:dsize => [1; hiddendim; hiddendim; 1]
				),
			# kwargs
			:kwargs => Dict(
				:lambda => 0.5, # replaced in training
				:threshold => 0, # useless
				:contamination => 0, # useless
				:batchsize => 0, # replaced in training
				:iterations => 10000,
				:cbit => 1000,
				:nepochs => nothing,
				:verbfit => verbfit,
				# input functions like this, they are evaluated later
				:activation => :(Flux.relu),
				:layer => :(Flux.Dense),
				:tracked => true,
				:rdelta => 1e-5,
				:pz => :(randn),
				:alpha => 0, # to be iterated over,
				:eta => 0.001
				)
			),
		# this is going to be iterated over for the fit function
		:ffparams => IterTools.product([:batchsize => i for i in batchsizes],
							 [:alpha => i for i in [0; [10.0^i for i in -4:2:4]]]),
#							 [:alpha => i for i in [0]]),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => IterTools.product([:lambda => i for i in linspace(0,1,6)]),
		# the model constructor
		:model => AnomalyDetection.fmGANmodel,
		# model fit function
		:ff => AnomalyDetection.fit!,
		# anomaly score function
		:asf => AnomalyDetection.anomalyscore
		)
)

	### sigmaVAE ###
PARAMS[:sigmaVAE] = copy(PARAMS[:VAE])
PARAMS[:sigmaVAE][:mparams][:kwargs][:variant] = :(_sigma())

if isoforest
	PARAMS[:IsoForest] = 
	    Dict(
		# this is going to serve as model construction params and also to be saved
		# in io
		:mparams => Dict(
			# args for the model constructor, must be in correct order
			:args => DataStructures.OrderedDict( 
				:n_estimators => 500,
				:max_samples => "auto",
				:contamination => 0, 
				:max_features => 1.0, # useless
				:bootstrap => false, # useless
				:njobs => 1,
				:verbose => 0, # replaced in training
				), 
			# kwargs
			:kwargs => Dict(
				)
			),
		# this is going to be iterated over for the fit function
		:ffparams => IterTools.product(),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => IterTools.product(),
		# the model constructor
		:model => IsolationForest,
		# model fit function
		:ff => fit!,
		# anomaly score function
		:asf => anomalyscore
		)
end