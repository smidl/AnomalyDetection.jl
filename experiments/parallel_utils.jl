###########################
### general NN settings ###
###### run settings #######
const hiddendim = 16
const latentdim = 8
const activation = Flux.relu
const verbfit = false
const batchsizes = [20, 256]
###############################

"""
	save_io(path, params, ascore, labels, algo_name)

Saves an algorithm output and input - params and anomaly scores.
"""
function save_io(path, file, mparams, trascore, trlabels, tstascore, tstlabels, loss, 
	alg_label, nnparams)
	mkpath(path)
	FileIO.save(joinpath(path, file), "params", mparams, "training_anomaly_score", trascore,
		"training_labels", trlabels, "testing_anomaly_score", tstascore, 
		"testing_labels", tstlabels, "loss", loss, "algorithm", alg_label, 
		"NN_params", nnparams)   
end

# model-specific saving routines
save_io(path, file, m, mparams, tras, trl, tstas, tstl) =
	save_io(path, file, mparams, tras, trl, tstas, tstl, nothing, string(m), nothing)
save_io(path, file, m::AnomalyDetection.kNN, mparams, tras, trl, tstas, tstl) =
	save_io(path, file, mparams, tras, trl, tstas, tstl, nothing, string(m), nothing)
save_io{model<:AnomalyDetection.genmodel}(path, file, m::model, mparams, tras, trl, tstas, tstl, alg) =
	save_io(path, file, mparams, tras, trl, tstas, tstl, m.history, string(m),
		map(Flux.Tracker.data, Flux.params(m)))

"""
	get_data(dataset_name, iteration)

Returns training and testing dataset.
"""
function get_data(dataset_name, iteration)
	# settings
	# ratio of training to all data
	alpha = 0.8 
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
	# random seed 
	seed = Int64(iteration)

	# this might fail for url
	basicset = AnomalyDetection.Basicset(joinpath(loda_path, dataset_name))
	trdata, tstdata, clusterdness = AnomalyDetection.makeset(basicset, alpha, difficulty, frequency, variation,
		seed = seed)
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
	train(dataset_name, iteration, alg) 

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

	experiment(data, tp[:model], tp[:mparams], tp[:ff], tp[:ffparams],
		tp[:asf], tp[:asfparams], path, alg)

	println("Training of $alg on $path finished!")
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
		ff(model, ndata)

		# inner loop over anomaly score parameters
		for args in asfp
			# update model parameters, filename and save actual values to mparams
			__fname = updateparams!(model, _fname, args, mparams)

			# get anomaly scores
			tras, tstas = getas(data, model, asf)
			
			# save input and output
			__fname = string(__fname, ".jld")
			save_io(outpath, __fname, model, mparams, tras, trdata.labels, tstas, 
				tstdata.labels)
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
		mparams[:args][a[1]] = a[2]
	end
	return fname
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
	topparams[:asfparams] = product([:k => i for i in Int.(round.(linspace(1, 2*sqrt(trN), 5)))])
end

function dataparams!{model<:AnomalyDetection.genmodel}(m::Type{model}, topparams, data)
	# change the esize and dsize params based on data size
	indim, trN = size(data[1].data[:,data[1].labels.==0])
	topparams[:mparams][:args][:esize][1] = indim
	topparams[:mparams][:args][:dsize][end] = indim

	# also modify the batchsize if it is larger than the dataset size
	# this is a little awkward but universal
	# steps: 
	# 1) if there is an L larger than datasize trN, create a new pair :L => trN
	# 2) filter out those pairs where :L > trN
	# 3) remove duplicates (if there are more pairs with :L > trN)
	ls = Array{Any,1}([x for x in topparams[:ffparams].xss])
	for l in ls
		map(x -> ((x[1]==:L && x[2] > trN)? push!(l, (x[1] => trN)) : (nothing)), l)
	end

	for i in 1:size(ls,1)
	 	ls[i] = filter(x -> !(x[1]==:L && x[2] > trN), ls[i])
	    ls[i] = unique(ls[i])
	end
	topparams[:ffparams] = product(ls...)
end

const PARAMS = Dict(
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
				:reduced_dim => true
				)
			),
		# this is going to be iterated ver for the fit function
		:ffparams => product(),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => product([:k => i for i in [1, 3, 5, 11, 21]]),
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
				:dsize => [latentdim; hiddendim; hiddendim; 1],
				:L => 0, # replaced in training
				:threshold => 0, # useless
				:contamination => 0, # useless
				:iterations => 10000,
				:cbit => 10000,
				:verbfit => verbfit
				), 
			# kwargs
			:kwargs => Dict(
				# input functions like this, they are evaluated later (only here)
				:activation => :(Flux.relu),
				:layer => :(Flux.Dense),
				:tracked => true,
				:rdelta => 1e-5
				)
			),
		# this is going to be iterated over for the fit function
		:ffparams => product([:L => i for i in batchsizes]),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => product(),
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
				:dsize => [latentdim; hiddendim; hiddendim; 1],
				:lambda => 0, # replaced in training
				:threshold => 0, # useless
				:contamination => 0, # useless
				:iterations => 10000,
				:cbit => 10000,
				:verbfit => verbfit,
				:L => 0 # replaced in training
				), 
			# kwargs
			:kwargs => Dict(
				:M => 1,
				# input functions like this, they are evaluated later
				:activation => :(Flux.relu), 
				:layer => :(Flux.Dense),
				:tracked => true,
				:rdelta => 1e-5
				)
			),
		# this is going to be iterated over for the fit function
		:ffparams => product([:L => i for i in batchsizes], 
#							[:lambda => i for i in [10.0^i for i in 0:-1:-4]]),
							[:lambda => i for i in [10.0^i for i in -4]]),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => product(),
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
				:dissize => [1 + latentdim; hiddendim; hiddendim; 1],
				:lambda => 0, # replaced in training
				:threshold => 0, # useless
				:contamination => 0, # useless
				:iterations => 10000,
				:cbit => 10000,
				:verbfit => verbfit,
				:L => 0 # replaced in training
				), 
			# kwargs
			:kwargs => Dict(
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
		:ffparams => product([:L => i for i in batchsizes], 
#							[:lambda => i for i in [0.0; [10.0^i for i in -4:2:4]]]),
							[:lambda => i for i in [1.0]]),
		# this is going to be iterated over for the anomalyscore function
		:asfparams => product([:alpha => i for i in linspace(0,1,6)]),
		# the model constructor
		:model => AnomalyDetection.sVAEmodel,
		# model fit function
		:ff => AnomalyDetection.fit!,
		# anomaly score function
		:asf => AnomalyDetection.anomalyscore
		)
)
