
using Flux
using JLD
using PyPlot

code_path = "../src"

push!(LOAD_PATH, code_path)
using AnomalyDetection

# load data
dataset = load("toy_data_3.jld")["data"]
x = dataset.data[:,dataset.labels.==0]

# VAE settings
indim = size(x,1)
hiddendim = 10
latentdim = 2
nlayers = 2

# setup the VAE object
lambda = 1.0
L = 100 # samples for classification
# arguments: 4 problem dimensions, 
# predict threshold, contamination level, iterations, throttle, verbal fit
esize = [indim; hiddendim; hiddendim; latentdim*2]
dsize = [latentdim; hiddendim; hiddendim; indim]
model = VAEmodel(esize, dsize, lambda, 0, 0.1, 50, 1, true, L, activation = Flux.relu)
#model = VAEmodel(indim, hiddendim, latentdim, nlayers, lambda, 0, 0.1, 50, 1, true, L)

# fit the model
model.lambda = 0.001
model.verbfit = false
while AnomalyDetection.rerr(model, x).data[1] > 0.001
    AnomalyDetection.fit!(model, x)
    AnomalyDetection.evalloss(model, x)
end

model(x)

x

AnomalyDetection.mu(model, x)

AnomalyDetection.sigma(model,x)

AnomalyDetection.sample_z(model, x)

# predict labels
X = dataset.data
y = dataset.labels
model.contamination = size(y[y.==1],1)/size(y, 1)
tryhat = AnomalyDetection.predict(model, X)

model.verbfit = false
tryhat, tstyhat = AnomalyDetection.quickvalidate!(dataset, dataset, model)

using ScikitLearn.Utils: meshgrid

# plot heatmap of the fit
figure()
title("classification results")
scatter(X[1, tstyhat.==1], X[2, tstyhat.==1], c = "r")
ax = gca()
ylim = ax[:get_ylim]()
xlim = ax[:get_xlim]()
xx, yy = meshgrid(linspace(xlim[1], xlim[2], 30), linspace(ylim[1], ylim[2], 30))
zz = zeros(size(xx))
for i in 1:size(xx, 1)
    for j in 1:size(xx, 2)
        zz[i,j] = AnomalyDetection.rerr(model, [xx[i,j], yy[i,j]]).data[1]
    end
end
contourf(xx, yy, zz)
scatter(X[1, tstyhat.==1], X[2, tstyhat.==1], c = "r")
scatter(X[1, tstyhat.==0], X[2, tstyhat.==0], c = "g")
b = AnomalyDetection.generate_sample(model)
scatter(b[1], b[2], c = "y", label = "generated sample")
legend()
show()

# what are the codes?
figure()
title("code distribution")
z1 = model.vae.encoder(X[:,1:30]).data
z2 = model.vae.encoder(X[:,31:60]).data
z3 = model.vae.encoder(X[:,61:90]).data
za = model.vae.encoder(X[:,91:end]).data

scatter(z1[1,:], z1[2,:], label = "first cluster")
scatter(z2[1,:], z2[2,:], label = "second cluster")
scatter(z3[1,:], z3[2,:], label = "third cluster")
scatter(za[1,:], za[2,:], s = 10, label = "anomalous data")
legend()
show()
