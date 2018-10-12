
using PyPlot, FileIO, AnomalyDetection, EvalCurves, Flux
import PyPlot: plot
include("./plots.jl")

dataset = load("toy_data_3.jld2" )["data"]

X = AnomalyDetection.Float.(dataset.data)
Y = dataset.labels
nX = X[:, Y.==0]

# set problem dimensions
indim = size(X,1)
hiddendim = 4
latentdim = 2
nlayers = 3

# model constructor parameters
esize = [indim; hiddendim; hiddendim; latentdim]; # encoder architecture
dsize = [latentdim; hiddendim; hiddendim; indim]; # decoder architecture
batchsize = 30 # batchsize
threshold = 0 # classification threshold, is recomputed when calling fit!
contamination = size(Y[Y.==1],1)/size(Y,1) # to set the decision threshold
iterations = 5000
cbit = 1000 # when callback is printed
nepochs = Int(ceil(batchsize*iterations/size(nX,2))) # if this is supplied, do epoch training
verbfit = true 
activation = Flux.relu
rdelta = 0.002 # reconstruction error threshold when training is stopped
Beta = 1.0 # for automatic threshold computation, in [0, 1] 
# 1.0 = tight around normal samples
tracked = true # do you want to store training progress?
# it can be later retrieved from model.history
eta = 0.0001

# model might have to be restarted if loss is > 0.01
model = AEmodel(esize, dsize; batchsize = batchsize, threshold=threshold, 
    contamination=contamination, iterations=iterations, cbit=cbit, 
    nepochs = nepochs,
    verbfit=verbfit, activation = activation, rdelta = rdelta, 
    tracked = tracked, eta = eta)

AnomalyDetection.fit!(model, nX)
AnomalyDetection.setthreshold!(model, X)
AnomalyDetection.evalloss(model, nX)

plot(model)
show()

model(nX)

nX

model(X)

X

# predict labels
yhat = AnomalyDetection.predict(model, X)

# training data = testing data
# this outputs labels
tryhat, tsthat, _, _ = AnomalyDetection.rocstats(dataset, dataset, model);

# anomaly score contour plot
# get limits of the figure
xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)

# compute the anomaly score on a grid
x = range(xl[1], stop=xl[2], length=30)
y = range(yl[1], stop=yl[2], length=30)
zz = zeros(size(y,1),size(x,1))
for i in 1:size(y, 1)
    for j in 1:size(x, 1)
        zz[i,j] = AnomalyDetection.anomalyscore(model, AnomalyDetection.Float.([x[j], y[i]]))
    end
end

# plot it all
f = figure()
contourf(x, y, zz)
scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = "r", 
    label = "predicted positive")
scatter(X[1, tryhat.==0], X[2, tryhat.==0], c = "g", 
    label = "predicted negative")
title("classification results")
xlim(xl)
ylim(yl)
legend()
show()

# what are the codes?
z1 = model.ae.encoder(X[:,1:30]).data
z2 = model.ae.encoder(X[:,31:60]).data
z3 = model.ae.encoder(X[:,61:90]).data
za = model.ae.encoder(X[:,91:end]).data

figure()
title("code distribution")
scatter(z1[1,:], z1[2,:], label = "first cluster")
scatter(z2[1,:], z2[2,:], label = "second cluster")
scatter(z3[1,:], z3[2,:], label = "third cluster")
scatter(za[1,:], za[2,:], s = 3, label = "anomalous data")
legend()
show()

# plot ROC curve and compute AUROC score
ascore = AnomalyDetection.anomalyscore(model, X);
fprvec, tprvec = EvalCurves.roccurve(ascore, Y)
auroc = round(EvalCurves.auc(fprvec, tprvec),digits=3)
EvalCurves.plotroc((fprvec, tprvec, "AUROC = $(auroc)"))
show()
