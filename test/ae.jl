# input data setup
esize = [xdim; hiddendim; latentdim]
dsize = [latentdim; hiddendim; xdim]
L = 5

@testset "AE" begin
	net = AE(esize, dsize,
		activation = Flux.relu, layer = Flux.Dense)
	@test size(net.encoder.layers,1) == 2
	@test size(net.decoder.layers,1) == 2
	@test size(net.encoder(X)) == (latentdim, N)
	rX = net(X)
	@test size(rX) == (xdim,N)
	@test typeof(rX) <: Flux.TrackedArray{AnomalyDetection.Float,2}
	l = AnomalyDetection.loss(net,nX)
	@test size(l) == ()
	history = ValueHistories.MVHistory()
	AnomalyDetection.fit!(net, nX, L, verb = false, history = history)
	AnomalyDetection.fit!(net, nX, L, iterations=100, cbit = 100, rdelta = Inf, verb= false)
	@test size(get(history,:loss)[1],1) == 1000
	@test l > AnomalyDetection.loss(net,nX)
	@test typeof(AnomalyDetection.anomalyscore(net, X[:,1])) <: AnomalyDetection.Float
	ascore = AnomalyDetection.anomalyscore(net, X)
	@test typeof(ascore) <: Array{AnomalyDetection.Float,1}
	@test findmax(ascore)[2] == N
	labels = AnomalyDetection.classify(net, X, 2.0)
	@test size(labels,1) == N
	@test labels[end] == 1
	@test minimum(labels[1:end-1] .== 0)
	sort!(ascore)
	@test typeof(AnomalyDetection.getthreshold(net, X, 0.1)) == AnomalyDetection.Float
	@test AnomalyDetection.getthreshold(net, X, 0.1) == ascore[end-1]
	@test AnomalyDetection.getthreshold(net, X, 0.0) == ascore[end]
	@test AnomalyDetection.getthreshold(net, X, 0.1, Beta = 0.5) == (ascore[end-1]+ascore[end])/2

	# test the classification model as well
	model = AEmodel(esize, dsize, batchsize = 5, threshold = .0, contamination = 0.1, 
		iterations = 1000, cbit = 100, verbfit = false, layer = Flux.Dense,
		activation = Flux.relu, rdelta = Inf, Beta = 1.0, tracked = true,
		eta = 0.0001)	
	rX = model(X)
	@test size(rX) == (xdim,N)
	@test typeof(rX) <: Flux.TrackedArray{AnomalyDetection.Float,2}
	l = AnomalyDetection.loss(model, nX)
	@test size(l) == ()
	AnomalyDetection.fit!(model, nX)
	@test size(get(model.history,:loss)[1],1) == 1000
	@test l > AnomalyDetection.loss(model,nX)
	AnomalyDetection.setthreshold!(model, X)
	@test model.contamination == 1/N
	ascore = AnomalyDetection.anomalyscore(model, X)
	@test typeof(ascore) <: Array{AnomalyDetection.Float,1}
	@test findmax(ascore)[2] == N
	labels = AnomalyDetection.classify(model, X)
	@test size(labels,1) == N
	@test labels[end] == 1
	@test minimum(labels[1:end-1] .== 0)
	sort!(ascore)
	@test ascore[end-1] == model.threshold
end 