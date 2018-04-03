###########################
### general NN settings ###
###### run settings #######
run_settings = Dict(
	"hiddendim" => 32,
	"latentdim" => 16,
	"activation" => Flux.relu,
	"verbfit" => false,
	"batchsizes" => [512]
	)

### precompilation settings ###
comp_settings = Dict(
	"hiddendim" => 2,
	"latentdim" => 1,
	"activation" => Flux.relu,
	"verbfit" => false,
	"batchsizes" => [1]
	)
###############################

"""
	prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed, [repetition, verb])

Prepare a single experiment.
"""
function prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed; 
	repetition = 0, verb = false)
	for i in 1:max(repetition,1)
		# load basic dataset and call makeset to extract testing and training data
		basicset = Basicset(joinpath(loda_path, dataset_name))
		trdata, tstdata, clusterdness = AnomalyDetection.makeset(basicset, alpha, difficulty, frequency, variation,
			seed = seed)

		# now save it
		fname = joinpath(export_path, string(lpad(size(trdata.data,1), 4, 0), "_", dataset_name, 
			"_", alpha, "_", difficulty, "_", frequency, "_", variation))
		# if more samples are requested, create subfolders
		if repetition > 0
			fname = joinpath(fname, "$i")
		end
		mkpath(fname)
		save(joinpath(fname, "training_data.jld"), "dataset", trdata)
		save(joinpath(fname, "testing_data.jld"), "dataset", tstdata)

		if verb
			println("Data saved to ", fname)
		end
	end
end

"""
	prepare_experiment_data(repetition)

Set up data for the Loda experiment.
"""
function prepare_experiment_data(repetition)
	# settings
	# ratio of training to all data
	alpha = 0.8 
	# easy/medium/hard/very_hard problem based on similarity of anomalous measurements to normal
	difficulty = "easy" 
	# ratio of anomalous to normal data
	frequency = 0.05 
	# low/high - should anomalies be clustered or not
	variation = "low"
	# random seed 
	seed = false 
	# verbosity of the saving routine
	verb = false

	files = readdir(Experiments.loda_path)

	# export the datasets
	for dataset_name in files
		try
			Experiments.prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed,
				repetition = repetition, verb = verb)
		end
	end

	# now export those datasets that dont have easy difficulty anomalies
	# you can check them using "julia print_loda_overview.jl"
	difficulty = "medium"

	for dataset_name in ["madelon", "gisette"]
		Experiments.prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed,
				repetition = repetition, verb = verb)
	end

	difficulty = "hard"
	for dataset_name in ["vertebral-column"]
		Experiments.prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed,
				repetition = repetition, verb = verb)
	end
end

"""
	run_experiment(dpaths, mode)

Train all the algorithms on data in folders given by data paths.
If mode == "compile", it will run the function with dummy settings to precompile it.
"""
function run_experiment(dpaths, mode)
	@parallel for dpath in dpaths
		subpaths = readdir(dpath) # list all subfolders
		for subpath in joinpath.(dpath,subpaths)
			# it may not (but should) be a proper subpath
			if !isdir(subpath)
				continue
			end
			@parallel for f in [trainAE, trainVAE, trainsVAE, trainGAN, trainfmGAN, trainkNN]
				f(subpath, mode)	
			end
		end
	end
end

"""
	save_io(path, params, ascore, labels, algo_name)

Saves an algorithm output and input - params and anomaly scores.
"""
function save_io(path, params, ascore, labels, loss, algo_name)
	mkpath(path)
	save(joinpath(path,"io.jld"), "params", params, "anomaly_score", ascore, 
		"labels", labels, "loss", loss, "algorithm", algo_name)   
end

##########
### AE ###
##########

### over this set of parameter values the AE will be trained for each dataset ###

"""
	trainAE(path, mode)

Trains an autoencoder.
"""
function trainAE(path, mode)
	# load data
	trdata = load(joinpath(path, "training_data.jld"))["dataset"]
	tstdata = load(joinpath(path, "testing_data.jld"))["dataset"]
	trX = trdata.data;
	trY = trdata.labels;
	tstX = tstdata.data;
	tstY = tstdata.labels;
	indim, trN = size(trX[:,trY.==0])

	# precompilation
	if mode == "run"
		settings = run_settings
	else
		settings = comp_settings
	end

	hiddendim = settings["hiddendim"]
	latentdim = settings["latentdim"]
	batchsizes = settings["batchsizes"]
	verbfit = settings["verbfit"]
	activation = settings["activation"]

	# over these parameters will be iterated
	if mode == "run"
		AEparams = Dict(
			"L" => batchsizes # batchsize
			)
	else
		AEparams = Dict(
			"L" => batchsizes # batchsize
			)
	end
	
	# set params to be saved later
	params = Dict(
			# set problem dimensions
		"indim" => indim,
		"hiddendim" => hiddendim,
		"latentdim" => latentdim,
		# model constructor parameters
		"esize" => [indim; hiddendim; hiddendim; latentdim], # encoder architecture
		"dsize" => [latentdim; hiddendim; hiddendim; indim], # decoder architecture
		"L" => 0, # batchsize, will be iterated over
		"threshold" => 0, # classification threshold, is recomputed when calling fit!
		"contamination" => size(trY[trY.==1],1)/size(trY[trY.==0],1), # to set the decision threshold
		"iterations" => 5000,
		"cbit" => 1000, # when callback is printed
		"verbfit" => verbfit, 
		"activation" => string(activation),
		"rdelta" => 1e-5, # reconstruction error threshold when training is stopped
		"Beta" => 1.0, # for automatic threshold computation, in [0, 1] 
		# 1.0 = tight around normal samples
		"tracked" => true # do you want to store training progress?
		# it can be later retrieved from model.traindata
		)

	# also, if batchsize is too large, add a batchsize param of the data size
	poplast = false
	if minimum(AEparams["L"]) > trN
		push!(AEparams["L"], trN)
		poplast = true
	end

	for L in AEparams["L"]
		if L > size(trX,2)
			continue
		end
		params["L"] = L
		# setup the model
		model = AEmodel(params["esize"], params["dsize"], params["L"], params["threshold"], 
			params["contamination"], params["iterations"], params["cbit"], params["verbfit"], 
			activation = activation, rdelta = params["rdelta"], tracked = params["tracked"],
			Beta = params["Beta"])
		# train the model
		AnomalyDetection.fit!(model, trX, trY)
		# get anomaly scores on testing data
		ascore = [Flux.Tracker.data(AnomalyDetection.anomalyscore(model, tstX[:,i]))
    		for i in 1:size(tstX,2)];
    	if mode == "run"
	    	# save anomaly scores, labels and settings
	    	pname = joinpath(path, string("AE_", L))
	    	save_io(pname, params, ascore, tstY, model.traindata, "AE")
	    end
	end

	# delete the last element of the 
	if poplast
		pop!(AEparams["L"])
	end

	println("AE training on $(path) finished!")
end

###########
### VAE ###
###########	

"""
	trainVAE(path, mode)

Trains a VAE and classifies training data in path..
"""
function trainVAE(path, mode)
	# load data
	trdata = load(joinpath(path, "training_data.jld"))["dataset"]
	tstdata = load(joinpath(path, "testing_data.jld"))["dataset"]
	trX = trdata.data;
	trY = trdata.labels;
	tstX = tstdata.data;
	tstY = tstdata.labels;
	indim, trN = size(trX[:,trY.==0])

	# precompilation
	if mode == "run"
		settings = run_settings
	else
		settings = comp_settings
	end

	hiddendim = settings["hiddendim"]
	latentdim = settings["latentdim"]
	batchsizes = settings["batchsizes"]
	verbfit = settings["verbfit"]
	activation = settings["activation"]

	# over these parameters will be iterated
	if mode == "run"
		VAEparams = Dict(
			"L" => batchsizes,
			"lambda" => [10.0^i for i in 0:-1:-4]
			)
	else
		VAEparams = Dict(
			"L" => batchsizes,
			"lambda" => [0.0]
			)
	end

	# also, if batchsize is too large, add a batchsize param of the data size
	poplast = false
	if minimum(VAEparams["L"]) > trN
		push!(VAEparams["L"], trN)
		poplast = true
	end

	# set params to be saved later
	params = Dict(
		# set problem dimensions
		"indim" => indim,
		"hiddendim" => hiddendim,
		"latentdim" => latentdim,
		# model constructor parameters
		"esize" => [indim; hiddendim; hiddendim; latentdim*2], # encoder architecture
		"dsize" => [latentdim; hiddendim; hiddendim; indim], # decoder architecture
		"lambda" => 1, # KLD weight in loss function
		"L" => 0, # batchsize, will be iterated over
		"threshold" => 0, # classification threshold, is recomputed when calling fit!
		"contamination" => size(trY[trY.==1],1)/size(trY[trY.==0],1), # to set the decision threshold
		"iterations" => 10000,
		"cbit" => 5000, # when callback is printed
		"verbfit" => verbfit, 
		"M" => 1, # number of samples for reconstruction error, set higher for classification
		"activation" => string(activation),
		"rdelta" => 1e-5, # reconstruction error threshold when training is stopped
		"Beta" => 1.0, # for automatic threshold computation, in [0, 1] 
		# 1.0 = tight around normal samples
		"tracked" => true # do you want to store training progress?
		# it can be later retrieved from model.traindata
		)
	
	for L in VAEparams["L"], lambda in VAEparams["lambda"]
		if L > trN
			continue
		end
		params["L"] = L
		params["lambda"] = lambda

		# setup the model
		model = VAEmodel(params["esize"], params["dsize"], params["lambda"],	params["threshold"], 
			params["contamination"], params["iterations"], params["cbit"], params["verbfit"],
			params["L"], M = params["M"], activation = activation, rdelta = params["rdelta"], 
			Beta = params["Beta"], tracked = params["tracked"])
		# train the model
		AnomalyDetection.fit!(model, trX, trY)
		# get anomaly scores on testing data
		params["M"] = 10
		model.M = params["M"] # set higher for stable classification
		ascore = [Flux.Tracker.data(AnomalyDetection.anomalyscore(model, tstX[:,i]))
    		for i in 1:size(tstX,2)];
    	if mode == "run"
	    	# save anomaly scores, labels and settings
	    	pname = joinpath(path, string("VAE_$(L)_$(lambda)"))
	    	save_io(pname, params, ascore, tstY, model.traindata, "VAE")
	    end
	end

	# delete the last element of the 
	if poplast
		pop!(VAEparams["L"])
	end

	println("VAE training on $(path) finished!")
end

############
### sVAE ###
############	

"""
	trainsVAE(path, mode)

Trains a sVAE and classifies training data in path..
"""
function trainsVAE(path, mode)
	# load data
	trdata = load(joinpath(path, "training_data.jld"))["dataset"]
	tstdata = load(joinpath(path, "testing_data.jld"))["dataset"]
	trX = trdata.data;
	trY = trdata.labels;
	tstX = tstdata.data;
	tstY = tstdata.labels;
	indim, trN = size(trX[:,trY.==0])

	# precompilation
	if mode == "run"
		settings = run_settings
	else
		settings = comp_settings
	end

	hiddendim = settings["hiddendim"]
	latentdim = settings["latentdim"]
	batchsizes = settings["batchsizes"]
	verbfit = settings["verbfit"]
	activation = settings["activation"]

	# over these parameters will be iterated
	if mode == "run"
		sVAEparams = Dict(
		"L" => batchsizes,
		"lambda" => push!([10.0^i for i in -2:2], 0.0), # data fit error term in loss
		"alpha" => linspace(0,1,5) # data fit error term in anomaly score
		)

	else
		sVAEparams = Dict(
			"L" => batchsizes,
			"lambda" => [0.0], # data fit error term in loss
			"alpha" => [0.5] # data fit error term in anomaly score
			)
	end

	# set params to be saved later
	params = Dict(
		# set problem dimensions
		"indim" => indim,
		"hiddendim" => hiddendim,
		"latentdim" => latentdim,
		# model constructor parameters
		"ensize" => [indim; hiddendim; hiddendim; latentdim*2], # encoder architecture
		"decsize" => [latentdim; hiddendim; hiddendim; indim], # decoder architecture
		"dissize" => [indim + latentdim; hiddendim; hiddendim; 1], # discriminator architecture
		"lambda" => 1, # data error weight for training
		"threshold" => 0, # classification threshold, is recomputed when calling fit!
		"contamination" => size(trY[trY.==1],1)/size(trY[trY.==0],1), # to set the decision threshold
		"iterations" => 10000,
		"cbit" => 5000, # when callback is printed
		"verbfit" => verbfit, 
		"L" => 0, # batchsize, will be iterated over
		"M" => 1, # number of samples for reconstruction error, set higher for classification
		"activation" => string(activation),
		"rdelta" => 1e-5, # reconstruction error threshold when training is stopped
		"alpha" => 0.5, # data error term for classification
		"Beta" => 1.0, # for automatic threshold computation, in [0, 1] 
		# 1.0 = tight around normal samples
		"tracked" => true, # do you want to store training progress?
		# it can be later retrieved from model.traindata
		"xsigma" => 1.0 # static estimate of data variance
		)

	# also, if batchsize is too large, add a batchsize param of the data size
	poplast = false
	if minimum(sVAEparams["L"]) > trN
		push!(sVAEparams["L"], trN)
		poplast = true
	end
	
	for L in sVAEparams["L"], lambda in sVAEparams["lambda"]
		if L > trN
			continue
		end
		params["L"] = L
		params["lambda"] = lambda

		# setup the model
		model = sVAEmodel(params["ensize"], params["decsize"], params["dissize"],
		 params["lambda"],	params["threshold"], params["contamination"], 
		 params["iterations"], params["cbit"], params["verbfit"], params["L"], 
		 M = params["M"], activation = activation, rdelta = params["rdelta"], 
			tracked = params["tracked"], Beta = params["Beta"], xsigma = params["xsigma"])
		# train the model
		AnomalyDetection.fit!(model, trX, trY)
		# get anomaly scores on testing data
		params["M"] = 10
		model.M = params["M"] # set higher for stable classification
		for alpha in sVAEparams["alpha"]
			params["alpha"] = alpha
			model.alpha = alpha
			ascore = [Flux.Tracker.data(AnomalyDetection.anomalyscore(model, tstX[:,i]))
	    		for i in 1:size(tstX,2)];
	    	if mode == "run"
		    	# save anomaly scores, labels and settings
		    	pname = joinpath(path, string("sVAE_$(L)_$(lambda)_$(alpha)"))
		    	save_io(pname, params, ascore, tstY, model.traindata, "sVAE")
		    end
	    end
	end

	# delete the last element of batchsizes
	if poplast
	 	pop!(sVAEparams["L"])
	end

	println("sVAE training on $(path) finished!")
end

###########
### GAN ###
###########

"""
	trainGAN(path, mode)

Trains a GAN and classifies training data in path.
"""
function trainGAN(path, mode)
	# load data
	trdata = load(joinpath(path, "training_data.jld"))["dataset"]
	tstdata = load(joinpath(path, "testing_data.jld"))["dataset"]
	trX = trdata.data;
	trY = trdata.labels;
	tstX = tstdata.data;
	tstY = tstdata.labels;
	indim, trN = size(trX[:,trY.==0])
	
	# precompilation
	if mode == "run"
		settings = run_settings
	else
		settings = comp_settings
	end

	hiddendim = settings["hiddendim"]
	latentdim = settings["latentdim"]
	batchsizes = settings["batchsizes"]
	verbfit = settings["verbfit"]
	activation = settings["activation"]

	# over these parameters will be iterated
	if mode == "run"
		GANparams = Dict(
		"L" => batchsizes, # batchsize
		"lambda" => linspace(0,1,5) # weight of reconstruction error in anomalyscore
		)
	else
		GANparams = Dict(
		"L" => batchsizes, # batchsize
		"lambda" => [0.0] # weight of reconstruction error in anomalyscore
		)
	end

	# set params to be saved later
	params = Dict(
			# set problem dimensions
		"indim" => indim,
		"hiddendim" => hiddendim,
		"latentdim" => latentdim,
		# model constructor parameters
		"gsize" => [latentdim; hiddendim; hiddendim; indim], # generator architecture
		"dsize" => [indim; hiddendim; hiddendim; 1], # discriminator architecture
		"threshold" => 0, # classification threshold, is recomputed when calling fit!
		"contamination" => size(trY[trY.==1],1)/size(trY[trY.==0],1), # to set the decision threshold
		"lambda" => 0.5, # anomaly score rerr weight
		"L" => 0, # batchsize
		"iterations" => 10000,
		"cbit" => 5000, # when callback is printed
		"verbfit" => verbfit, 
		"pz" => string(randn),
		"activation" => string(activation),
		"rdelta" => 1e-5, # reconstruction error threshold when training is stopped
		"Beta" => 1.0, # for automatic threshold computation, in [0, 1] 
		# 1.0 = tight around normal samples
		"tracked" => true # do you want to store training progress?
		# it can be later retrieved from model.traindata
		)

	# also, if batchsize is too large, add a batchsize param of the data size
	poplast = false
	if minimum(GANparams["L"]) > trN
		push!(GANparams["L"], trN)
		poplast = true
	end

	for L in GANparams["L"]
		if L > trN
			continue
		end

		# setup the model
		model = GANmodel(params["gsize"], params["dsize"], params["lambda"], params["threshold"], 
			params["contamination"], L, params["iterations"], params["cbit"], 
			params["verbfit"], pz = randn, activation = activation, rdelta = params["rdelta"], 
			tracked = params["tracked"], Beta = params["Beta"])
		# train the model
		AnomalyDetection.fit!(model, trX, trY)
		for lambda in GANparams["lambda"]
			params["lambda"] = lambda
			model.lambda = lambda
			# get anomaly scores on testing data
			ascore = [Flux.Tracker.data(AnomalyDetection.anomalyscore(model, tstX[:,i]))
	    		for i in 1:size(tstX,2)];
	    	if mode == "run"
		    	# save anomaly scores, labels and settings
		    	pname = joinpath(path, string("GAN_$(L)_$(lambda)"))
	    		save_io(pname, params, ascore, tstY, model.traindata, "GAN")
	    	end
	    end
	end

	# delete the last element of batchsizes
	if poplast
		pop!(GANparams["L"])
	end

	println("GAN training on $(path) finished!")
end

#############
### fmGAN ###
#############


"""
	trainfmGAN(path, mode)

Trains a fmGAN and classifies training data in path..
"""
function trainfmGAN(path, mode)
	# load data
	trdata = load(joinpath(path, "training_data.jld"))["dataset"]
	tstdata = load(joinpath(path, "testing_data.jld"))["dataset"]
	trX = trdata.data;
	trY = trdata.labels;
	tstX = tstdata.data;
	tstY = tstdata.labels;
	indim, trN = size(trX[:,trY.==0])
	
	# precompilation
	if mode == "run"
		settings = run_settings
	else
		settings = comp_settings
	end

	hiddendim = settings["hiddendim"]
	latentdim = settings["latentdim"]
	batchsizes = settings["batchsizes"]
	verbfit = settings["verbfit"]
	activation = settings["activation"]

	# over these params will be iterated
	if mode == "run"
		fmGANparams = Dict(
		"L" => batchsizes, # batchsize
		"lambda" => linspace(0,1,5), # weight of reconstruction error in anomalyscore
		"alpha" => push!([10.0^i for i in -2:2], 0.0) 
		)
	else
		fmGANparams = Dict(
		"L" => batchsizes, # batchsize
		"lambda" => [0], # weight of reconstruction error in anomalyscore
		"alpha" => [0.5] 
		)
	end

	# set params to be saved later
	params = Dict(
			# set problem dimensions
		"indim" => indim,
		"hiddendim" => hiddendim,
		"latentdim" => latentdim,
		# model constructor parameters
		"gsize" => [latentdim; hiddendim; hiddendim; indim], # generator architecture
		"dsize" => [indim; hiddendim; hiddendim; 1], # discriminator architecture
		"threshold" => 0, # classification threshold, is recomputed when calling fit!
		"contamination" => size(trY[trY.==1],1)/size(trY[trY.==0],1), # to set the decision threshold
		"lambda" => 0.5, # anomaly score rerr weight
		"L" => 0, # batchsize
		"iterations" => 10000,
		"cbit" => 5000, # when callback is printed
		"verbfit" => verbfit, 
		"pz" => string(randn),
		"activation" => string(activation),
		"rdelta" => 1e-5, # reconstruction error threshold when training is stopped
		"alpha" => 0.5, # weight of discriminator score in generator loss training
		"Beta" => 1.0, # for automatic threshold computation, in [0, 1] 
		# 1.0 = tight around normal samples
		"tracked" => true # do you want to store training progress?
		# it can be later retrieved from model.traindata
		)

	# also, if batchsize is too large, add a batchsize param of the data size
	poplast = false
	if minimum(fmGANparams["L"]) > trN
		push!(fmGANparams["L"], trN)
		poplast = true
	end

	for L in fmGANparams["L"], alpha in fmGANparams["alpha"]
		if L > trN
			continue
		end
		
		# setup the model
		model = fmGANmodel(params["gsize"], params["dsize"], params["lambda"], params["threshold"], 
			params["contamination"], L, params["iterations"], params["cbit"], 
			params["verbfit"], pz = randn, activation = activation, rdelta = params["rdelta"], 
			tracked = params["tracked"], Beta = params["Beta"], alpha = alpha)
		# train the model
		AnomalyDetection.fit!(model, trX, trY)
		for lambda in fmGANparams["lambda"]
			params["lambda"] = lambda
			model.lambda = lambda
			# get anomaly scores on testing data
			ascore = [Flux.Tracker.data(AnomalyDetection.anomalyscore(model, tstX[:,i]))
	    		for i in 1:size(tstX,2)];
			if mode == "run"
		    	# save anomaly scores, labels and settings
		    	pname = joinpath(path, string("fmGAN_$(L)_$(lambda)_$(alpha)"))
		    	save_io(pname, params, ascore, tstY, model.traindata, "fmGAN")
		    end
	    end
	end

	# delete the last element of batchsizes
	if poplast
		pop!(fmGANparams["L"])
	end

	println("fmGAN training on $(path) finished!")
end

###########
### kNN ###
###########

"""
	trainkNN(path, mode)

Trains a kNN and classifies training data in path.
"""
function trainkNN(path, mode)
	# load data
	trdata = load(joinpath(path, "training_data.jld"))["dataset"]
	tstdata = load(joinpath(path, "testing_data.jld"))["dataset"]
	trX = trdata.data;
	trY = trdata.labels;
	tstX = tstdata.data;
	tstY = tstdata.labels;
	indim, trN = size(trX)
	
	# set params to be saved later
	params = Dict(
		"k" => 1,
		"metric" => string(Euclidean()),
		"weights" => "distance",
		"threshold" => 0.5,
		"reduced_dim" => true,
		)

	if mode == "run"
		kvec = Int.(round.(linspace(1, 2*sqrt(trN), 5)))
	else
		kvec = [1]
	end

	@parallel for k in kvec 
		params["k"] = k
		# setup the model
		model = kNN(params["k"], metric = Euclidean(), weights = params["weights"], 
			threshold = params["threshold"], reduced_dim = params["reduced_dim"])

		# train the model
		AnomalyDetection.fit!(model, trX, trY)
		# get anomaly scores on testing data
		ascore = [Flux.Tracker.data(AnomalyDetection.anomalyscore(model, tstX[:,i]))
    		for i in 1:size(tstX,2)];
    	if mode == "run"
	    	# save anomaly scores, labels and settings
	    	pname = joinpath(path, string("kNN_$(k)"))
	    	save_io(pname, params, ascore, tstY, Dict{Any, Any}(), "kNN")
	    end
	end

	println("kNN training on $(path) finished!")
end
