###########################
### general NN settings ###
###### run settings #######
hiddendim = 16
latentdim = 8
activation = Flux.relu
verbfit = false
batchsizes = [20, 256]
###############################

"""
	save_io(path, params, ascore, labels, algo_name)

Saves an algorithm output and input - params and anomaly scores.
"""
function save_io(path, file, mparams, trascore, trlabels, tstascore, tstlabels, loss, 
	algo_name, nnparams)
	mkpath(path)
	FileIO.save(joinpath(path, file), "params", mparams, "training_anomaly_score", trascore,
		"training_labels", trlabels, "testing_anomaly_score", tstascore, 
		"testing_labels", tstlabels, "loss", loss, "algorithm", algo_name, 
		"NN_params", nnparams)   
end

# model-specific saving routines
save_io(path, file, m, mparams, tras, trl, tstas, tstl, alg) =
	save_io(path, file, mparams, tras, trl, tstas, tstl, nothing, alg, nothing)
save_io(path, file, m::AnomalyDetection.kNN, mparams, tras, trl, tstas, tstl, alg) =
	save_io(path, file, mparams, tras, trl, tstas, tstl, nothing, alg, nothing)
save_io{model<:AnomalyDetection.genmodel}(path, file, m::model, mparams, tras, trl, tstas, tstl, alg) =
	save_io(path, file, mparams, tras, trl, tstas, tstl, m.history, alg, Flux.params(m))

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
function train(dataset_name, iteration, alg) 
	data = get_data(dataset_name, iteration)
	(trdata, tstdata) = data 

	# top level of the param tree
	tp = deepcopy(PARAMS[Symbol(alg)])

	# data will be saved here
	path = 	joinpath(export_path, string("$(dataset_name)/$(alg)/$(iteration)"))

	# upgrade params according to data size
	dataparams!(tp[:model], tp, data)

	# outer loop over fit parameters
	for fargs in tp[:fparams]
		# create the io file name
		ffname = alg
			
		# mparams is the dictionary of used parameters that will be saved
		mparams = deepcopy(tp[:mparams]) # so that the original params are not owerwritten
		# universal model constructor
		model = tp[:model]([p for p in values(mparams[:args])]...; 
			[p[1] => eval(p[2]) for p in mparams[:kwargs]]...)
		# SortedDict causes problems during load(), now the order does not matter anymore
		mparams[:args] = Dict(mparams[:args]) 

		# update model parameters, filename and save actual values to mparams
		ffname = updateparams!(model, ffname, fargs, mparams)

		# fit the model
		tp[:f](model, trdata.data, trdata.labels)

		# inner loop over anomaly score parameters
		for args in tp[:asparams]
			# update model parameters, filename and save actual values to mparams
			fname = updateparams!(model, ffname, args, mparams)

			# get anomaly scores
			tras, tstas = getas(data, model, tp[:as])
			
			# save input and output
			fname = string(fname, ".jld")
			save_io(path, fname, model, mparams, tras, trdata.labels, tstas, tstdata.labels, 
				alg)
		end
	end
	println("Training of $alg on $path finished!")
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
	topparams[:asparams] = product([:k => i for i in Int.(round.(linspace(1, 2*sqrt(trN), 5)))])
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
	ls = Array{Any,1}([x for x in topparams[:fparams].xss])
	for l in ls
		map(x -> ((x[1]==:L && x[2] > trN)? push!(l, (x[1] => trN)) : (nothing)), l)
	end

	for i in 1:size(ls,1)
	 	ls[i] = filter(x -> !(x[1]==:L && x[2] > trN), ls[i])
	    ls[i] = unique(ls[i])
	end
	topparams[:fparams] = product(ls...)
end

const PARAMS = Dict(
	### kNN ###
	:kNN => Dict(
		# this serves as model construction params and also to be saved
		# in io
		:mparams => Dict(
			# args for the model constructor, must be in correct order
			:args => DataStructures.OrderedDict( 
				:k => 1
				), 
			# kwargs
			:kwargs => Dict(
				:metric => :(Distances.Euclidean()),
				:weights => "distance",
				:threshold => 0.5,
				:reduced_dim => true	
				)
			),
		# this is going to be iterated ver for the fit function
		:fparams => product(),
		# this is going to be iterated over for the anomalyscore function
		:asparams => product([:k => i for i in [1, 3, 5, 11, 21]]),
		# the model constructor
		:model => AnomalyDetection.kNN,
		# model fit function
		:f => AnomalyDetection.fit!,
		# anomaly score function
		:as => AnomalyDetection.anomalyscore
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
				:iterations => 5000,
				:cbit => 1000,
				:verbfit => false
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
		:fparams => product([:L => i for i in batchsizes]),
		# this is going to be iterated over for the anomalyscore function
		:asparams => product(),
		# the model constructor
		:model => AnomalyDetection.AEmodel,
		# model fit function
		:f => AnomalyDetection.fit!,
		# anomaly score function
		:as => AnomalyDetection.anomalyscore
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
				:iterations => 5000,
				:cbit => 1000,
				:verbfit => false,
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
		:fparams => product([:L => i for i in batchsizes], 
#							[:lambda => i for i in [10.0^i for i in 0:-1:-4]]),
							[:lambda => i for i in [10.0^i for i in -4]]),
		# this is going to be iterated over for the anomalyscore function
		:asparams => product(),
		# the model constructor
		:model => AnomalyDetection.VAEmodel,
		# model fit function
		:f => AnomalyDetection.fit!,
		# anomaly score function
		:as => AnomalyDetection.anomalyscore
		)
)
