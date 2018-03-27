
using Flux
using JLD
using PyPlot
using ScikitLearn.Utils: meshgrid

code_path = "../src"

push!(LOAD_PATH, code_path)
using AnomalyDetection

# load data
dataset = load("toy_data_3.jld")["data"]
X = dataset.data
Y = dataset.labels
nX = X[:, Y.==0]
M, N = size(X)

# GAN settings
zdim = 1 # code dimension
xdim = M # dimension of data
hiddendim = 32  

# setup the GAN model object
gsize = [zdim; hiddendim; hiddendim; xdim] # generator layout
dsize = [xdim; hiddendim*2; hiddendim*2; 1] # discriminator layout
lambda = 0.5 # anomaly score parameter in [0, 1]
# 1 - ignores the discriminator score
# 0- ignores the reconstruction error score
threshold = 0 # classification threshold, is recomputed (getthreshold or when using fit!)
contamination = size(Y[Y.==1],1)/size(Y, 1) # contamination ratio
m = 30 # batchsize
iterations = 15000 # no of iterations
cbit = 5000 # when should output be printed
verbfit = true # if output should be produced
pz = randn # code distribution (rand should also work)
activation = Flux.leakyrelu # should work better than relu
rdelta = 1e-5 # stop training after this reconstruction error is achieved
# this parameter is basically useless in the case of GANs
Beta = 1.0 # for automatic threshold computation, in [0, 1] 
# 1.0 = tight around normal samples
tracked = true # do you want to store training progress?
# it can be later retrieved from model.traindata
model = GANmodel(gsize, dsize, lambda, threshold, contamination, m, iterations, cbit,
    verbfit, pz = pz, activation = activation, rdelta = rdelta, Beta = Beta, 
    tracked = tracked)

# fit the model
Z = model.gan.pz(zdim, size(nX,2))
AnomalyDetection.evalloss(model, nX, Z)
AnomalyDetection.fit!(model, X, Y)
AnomalyDetection.evalloss(model, nX, Z)

# plot model loss
plot(model)

# generate new data
Xgen = AnomalyDetection.generate(model, N)

# plot them
figure()
title("Generator results and discriminator contourplot")
scatter(Xgen[1,:], Xgen[2,:]) # first plot jsut to get axis limits
ax = gca()
#ylim = ax[:get_ylim]()
#xlim = ax[:get_xlim]()
xlim = (min(minimum(X[1,:]), minimum(Xgen[1,:])) - 0.05, 
    max(maximum(X[1,:]), maximum(Xgen[1,:])) + 0.05)
ylim = (min(minimum(X[2,:]), minimum(Xgen[2,:])) - 0.05, 
    max(maximum(X[2,:]), maximum(Xgen[2,:])) + 0.05)
xx, yy = meshgrid(linspace(xlim[1], xlim[2], 30), linspace(ylim[1], ylim[2], 30))
zz = zeros(size(xx))
for i in 1:size(xx, 1)
    for j in 1:size(xx, 2)
        zz[i,j] = model.gan.d([xx[i,j], yy[i,j]]).tracker.data[1]
    end
end
axsurf = ax[:contourf](xx, yy, zz)
cb = colorbar(axsurf, fraction = 0.05, shrink = 0.5, pad = 0.1)
scatter(X[1,:], X[2,:], c = "k", label = "original data")
scatter(Xgen[1,:], Xgen[2,:], c="r", label = "generated data")
legend()
show()

# predict labels
tryhat = AnomalyDetection.predict(model, X)

# get all the labels
tryhat, tstyhat, _, _ = AnomalyDetection.rocstats(dataset, dataset, model);

# plot heatmap of the fit
figure()
title("classification results")
scatter(X[1, tstyhat.==1], X[2, tstyhat.==1], c = "r")
ax = gca()
xlim = ax[:get_xlim]()
ylim = ax[:get_ylim]()
xx, yy = meshgrid(linspace(xlim[1], xlim[2], 30), linspace(ylim[1], ylim[2], 30))
zz = zeros(size(xx))
for i in 1:size(xx, 1)
    for j in 1:size(xx, 2)
        zz[i,j] = model.gan.d([xx[i,j], yy[i,j]]).tracker.data[1]
    end
end
axsurf = ax[:contourf](xx, yy, zz)
cb = colorbar(axsurf, fraction = 0.05, shrink = 0.5, pad = 0.1)

scatter(X[1, tstyhat.==1], X[2, tstyhat.==1], c = "r", label = "predicted positive")
scatter(X[1, tstyhat.==0], X[2, tstyhat.==0], c = "g", label = "predicted negative")
b = AnomalyDetection.generate(model)
scatter(b[1], b[2], c = "y", label = "generated sample")
legend(loc = "upper right")
show()

# plot EER for different settings of lambda
using MLBase: roc, correctrate, precision, recall, f1score, false_positive_rate, false_negative_rate
n = 21
lvec = linspace(0,1,n)
eervec = zeros(n)
for i in 1:n
    model.lambda = lvec[i]
    tryhat, tsthat, trroc, tstroc = AnomalyDetection.rocstats(dataset.data, dataset.labels,
        dataset.data, dataset.labels, model, verb = false)
    eervec[i] = (false_positive_rate(tstroc) + false_negative_rate(tstroc))/2
end

figure()
title("equal error rate vs lambda")
plot(lvec, eervec)
xlabel("lambda")
ylabel("EER")
show()
