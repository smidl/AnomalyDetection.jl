
using PyPlot
using JLD
using ScikitLearn: @sk_import, fit!, predict
using ScikitLearn.Utils: meshgrid 

code_path = "../src/"

push!(LOAD_PATH, code_path)
using AnomalyDetection

dataset = load("toy_data_3.jld")["data"]

figure()
X = dataset.data
y = dataset.labels
scatter(X[1, y.==1], X[2, y.==1])
scatter(X[1, y.==0], X[2, y.==0])
show()

# set problem dimensions
indim = size(X,1)
hiddendim = 4
latentdim = 2
nlayers = 3

# model constructor parameters
esize = [indim; hiddendim; hiddendim; latentdim]; # encoder architecture
dsize = [latentdim; hiddendim; hiddendim; indim]; # decoder architecture
threshold = 0 # classification threshold, is recomputed when calling fit!
contamination = size(y[y.==1],1)/size(y[y.==0],1) # to set the decision threshold
iterations = 5000
cbit = 1000 # when callback is printed
verbfit = true 
activation = Flux.relu
rdelta = 0.005 # reconstruction error threshold when training is stopped

# select only normal data
x = X[:,y .== 0]

# model might have to be restarted if loss is > 0.01
model = AEmodel(esize, dsize, threshold, contamination,
    iterations, cbit, verbfit, activation = activation, rdelta = rdelta)

AnomalyDetection.fit!(model, X)
AnomalyDetection.evalloss(model, X)

model(x)

x

model(X)

X

# predict labels
yhat = AnomalyDetection.predict(model, X)

# training data = testing data
# this outputs labels
tryhat, tsthat, _, _ = AnomalyDetection.rocstats(dataset, dataset, model);

using ScikitLearn.Utils: meshgrid

# plot heatmap of the fit
figure()
title("classification results")
scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = "r")
ax = gca()
ylim = ax[:get_ylim]()
xlim = ax[:get_xlim]()
xx, yy = meshgrid(linspace(xlim[1], xlim[2], 30), linspace(ylim[1], ylim[2], 30))
zz = zeros(size(xx))
for i in 1:size(xx, 1)
    for j in 1:size(xx, 2)
        zz[i,j] = AnomalyDetection.loss(model, [xx[i,j], yy[i,j]]).data[1]
    end
end
axsurf = ax[:contourf](xx, yy, zz)
cb = colorbar(axsurf, fraction = 0.05, shrink = 0.5, pad = 0.1)
scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = "r", label = "predicted positive")
scatter(X[1, tryhat.==0], X[2, tryhat.==0], c = "g", label = "predicted negative")
legend(loc = "upper right")
show()

# what are the codes?
figure()
title("code distribution")
z1 = model.ae.encoder(X[:,1:30]).data
z2 = model.ae.encoder(X[:,31:60]).data
z3 = model.ae.encoder(X[:,61:90]).data
za = model.ae.encoder(X[:,91:end]).data

scatter(z1[1,:], z1[2,:], label = "first cluster")
scatter(z2[1,:], z2[2,:], label = "second cluster")
scatter(z3[1,:], z3[2,:], label = "third cluster")
scatter(za[1,:], za[2,:], s = 10, label = "anomalous data")
legend()
