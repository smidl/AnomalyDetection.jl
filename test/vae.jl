# input data setup
srand(123)
L = 5
esize = [xdim; hiddendim; latentdim*2]
dsize = [latentdim; hiddendim; xdim]
lambda = 0.001
M = 1

@testset "VAE" begin
	net = VAE(esize, dsize, 
		activation = Flux.relu, layer = Flux.Dense)
	@test size(net.encoder.layers,1) == 2
	@test size(net.decoder.layers,1) == 2
	@test size(net.encoder(X)) == (latentdim*2, N)
	rX = net(X)
	@test size(rX) == (xdim,N)
	@test typeof(rX) <: Flux.TrackedArray{AnomalyDetection.Float,2}
	@test typeof(AnomalyDetection.mu(net, net.encoder(X))) <: Flux.TrackedArray{AnomalyDetection.Float,2}
	@test typeof(AnomalyDetection.sigma(net, net.encoder(X))) <: Flux.TrackedArray{AnomalyDetection.Float,2}
	@test typeof(AnomalyDetection.sample_z(net, net.encoder(X)))  <: Flux.TrackedArray{AnomalyDetection.Float,2}
	l = AnomalyDetection.loss(net,nX, M, lambda)
	@test size(l) == ()
	@test size(AnomalyDetection.KL(net,nX)) == ()
	@test size(AnomalyDetection.rerr(net, nX, M)) == ()
	history = ValueHistories.MVHistory()
	AnomalyDetection.fit!(net, nX, L, lambda = lambda, verb = false, history = history)
	AnomalyDetection.fit!(net, nX, L, M=M, iterations=100, cbit = 100, rdelta = Inf, verb= false)
	@test size(get(history,:loss)[1],1) == 1000
	@test l > AnomalyDetection.loss(net,nX,M,lambda)
	@test typeof(AnomalyDetection.anomalyscore(net, X[:,1], M)) <: AnomalyDetection.Float
	ascore = AnomalyDetection.anomalyscore(net, X, M)
	@test typeof(ascore) <: Array{AnomalyDetection.Float,1}
	@test findmax(ascore)[2] == N
	labels = AnomalyDetection.classify(net, X, 2.0, M)
	@test size(labels,1) == N
	@test labels[end] == 1
	@test minimum(labels[1:end-1] .== 0)
	sort!(ascore)
	@test typeof(AnomalyDetection.getthreshold(net, X, M, 0.1)) == AnomalyDetection.Float
	@test abs(AnomalyDetection.getthreshold(net, X, M, 0.1) - ascore[end-1]) < 1.0
	@test abs(AnomalyDetection.getthreshold(net, X, M, 0.0) - ascore[end]) < 7.0
	@test abs(AnomalyDetection.getthreshold(net, X, M, 0.1, Beta = 0.5) - (ascore[end-1]+ascore[end])/2) < 3.0

	# test the classification model as well
	model = VAEmodel(esize, dsize, lambda, 0.0, 0.1, 1000, 100, false, 5,
		activation = Flux.relu, layer = Flux.Dense, rdelta = Inf, Beta = 0.9, 
		tracked = true)	
	rX = model(X)
	@test size(rX) == (xdim,N)
	@test typeof(rX) <: Flux.TrackedArray{AnomalyDetection.Float,2}
	l = AnomalyDetection.loss(model, nX)
	@test size(l) == ()
	AnomalyDetection.fit!(model, nX)
	AnomalyDetection.setthreshold!(model, X)
	@test size(get(model.history,:loss)[1],1) == 1000
	@test l > AnomalyDetection.loss(model,nX)
	ascore = AnomalyDetection.anomalyscore(model, X)
	@test typeof(ascore) <: Array{AnomalyDetection.Float,1}
	@test findmax(ascore)[2] == N
	labels = AnomalyDetection.classify(model, X)
	@test size(labels,1) == N
	@test labels[end] == 1
	@test minimum(labels[1:end-1] .== 0)
	sort!(ascore)
	@test abs(ascore[end-1] - model.threshold) < 0.3
end 