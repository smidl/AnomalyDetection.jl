
using Plots
plotly()
import Plots: plot
clibrary(:Plots)
using JLD

code_path = "../src"

push!(LOAD_PATH, code_path)
using AnomalyDetection

# load data
dataset = load("toy_data_3.jld")["data"]
X = AnomalyDetection.Float.(dataset.data)
Y = dataset.labels;
nX = X[:,Y.==0]

# sVAE settings
indim = size(X,1)
hiddendim = 10
latentdim = 2
nlayers = 2
N = size(nX,2)

# setup the VAE object
ensize = [indim; hiddendim; hiddendim; latentdim*2] # encoder architecture
decsize = [latentdim; hiddendim; hiddendim; indim] # decoder architecture
dissize = [indim + latentdim; hiddendim; hiddendim; 1] # discriminator architecture
lambda = 0.0 # weight of the data error term
threshold = 0 # classification threshold, is recomputed during fit!()
contamination = size(Y[Y.==1],1)/size(Y, 1) # for automatic threshold computation
iterations = 10000
cbit = 5000 # after this number of iteratiosn, callback is printed
verbfit = true
L = 50 # batchsize
M = 1 #  number of samples of X in reconstruction error
activation = Flux.relu
layer = Flux.Dense
rdelta = 1e-4 # reconstruction error threshold for training stopping
alpha = 0.5 # weighs between reconstruction error and discriminator score for classification
# 0 = only reconstruction error, 1 = only discriminator score
Beta = 1.0 # for automatic threshold computation, in [0, 1] 
# 1.0 = tight around normal samples
tracked = true # do you want to store training progress?
# it can be later retrieved from model.traindata
xsigma = 1.0 # static estimate of data variance
model = sVAEmodel(ensize, decsize, dissize, lambda, threshold, contamination, iterations, cbit, verbfit, 
    L, M = M, activation = activation, rdelta = rdelta, Beta = Beta, xsigma = xsigma,
    tracked = tracked, layer = layer)

# fit the model
AnomalyDetection.evalloss(model, nX)
AnomalyDetection.fit!(model, nX)
AnomalyDetection.evalloss(model, nX)

"""
	plot(model)

Plot the model loss.
"""
function plot(model::sVAEmodel)
	# plot model loss
	if model.history == nothing
		println("No data to plot, set tracked = true before training.")
		return
	else
        p = plot(model.history[:discriminator_loss], title = "model loss", 
            label = "discriminator loss", 
            xlabel = "iteration", ylabel = "loss", 
            seriestype = :line, 
            markershape = :none)
        plot!(model.history[:vae_loss], label = "VAE loss",
            seriestype = :line, markershape = :none)
        plot!(model.history[:reconstruction_error], label = "reconstruction error",
            seriestype = :line, markershape = :none, 
            c = :green,
            title = "model loss")
        return p
    end
end

# plot model loss
display(plot(model))
if !isinteractive()
    gui()
end

# plot heatmap of the fit
xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)
p = scatter(X[1, Y.==0], X[2, Y.==0], c = :red, label = "data",
    xlims=xl, ylims = yl, title = "discriminator contours", 
    c=:green)
Xrec = Flux.Tracker.data(model(X[:, Y.==0]))
scatter!(Xrec[1,:], Xrec[2,:], label = "reconstructed")
Xgen = AnomalyDetection.generate(model, 30)
scatter!(Xgen[1,:], Xgen[2,:], label = "generated", c = :black, markersize = 3,
    legend = (0.1, 0.7))

x = linspace(xl[1], xl[2], 30)
y = linspace(yl[1], yl[2], 30)
zz = zeros(size(y,1),size(x,1))
for i in 1:size(y, 1)
    for j in 1:size(x, 1)
        _x = AnomalyDetection.Float.([x[j], y[i]])
        _z = AnomalyDetection.getcode(model, _x)
        zz[i,j] = Flux.Tracker.data(AnomalyDetection.discriminate(model, _x, _z))[1]
    end
end
contourf!(x, y, zz, c = :viridis)

display(p)
if !isinteractive()
    gui()
end


model(nX)

nX

AnomalyDetection.mu(model, nX)

AnomalyDetection.sigma(model, nX)

AnomalyDetection.sample_z(model, nX)

# predict labels
AnomalyDetection.setthreshold!(model, X)
model.M = 20 # number of samples - for classification higher is better (more stable)
tryhat = AnomalyDetection.predict(model, X)

# get the labels and roc stats
tryhat, tstyhat = AnomalyDetection.rocstats(dataset, dataset, model);

# plot heatmap of the fit
xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)
p = scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = :red, label = "predicted positive",
    xlims=xl, ylims = yl, title = "classification results and anomaly score contours")
scatter!(X[1, tryhat.==0], X[2, tryhat.==0], c = :green, label = "predicted negative",
    legend = (0.1, 0.7))

x = linspace(xl[1], xl[2], 30)
y = linspace(yl[1], yl[2], 30)
zz = zeros(size(y,1),size(x,1))
for i in 1:size(y, 1)
    for j in 1:size(x, 1)
        zz[i,j] = AnomalyDetection.anomalyscore(model, AnomalyDetection.Float.([x[j], y[i]]))
    end
end
contourf!(x, y, zz, c = :viridis)

display(p)
if !isinteractive()
    gui()
end

# what are the codes?

z1 = AnomalyDetection.getcode(model, X[:,1:30]).data
z2 = AnomalyDetection.getcode(model, X[:,31:60]).data
z3 = AnomalyDetection.getcode(model, X[:,61:90]).data
za = AnomalyDetection.getcode(model, X[:,91:end]).data

p = scatter(z1[1,:], z1[2,:], label = "first cluster")
scatter!(z2[1,:], z2[2,:], label = "second cluster")
scatter!(z3[1,:], z3[2,:], label = "third cluster")
scatter!(za[1,:], za[2,:], s = 10, label = "anomalous data")

display(p)
if !isinteractive()
    gui()
end

# plot the roc curve as well
ascore = AnomalyDetection.anomalyscore(model, X);
recvec, fprvec = AnomalyDetection.getroccurve(ascore, Y)

function plotroc(args...)
    # plot the diagonal line
    p = plot(linspace(0,1,100), linspace(0,1,100), c = :gray, alpha = 0.5, xlim = [0,1],
    ylim = [0,1], label = "", xlabel = "false positive rate", ylabel = "true positive rate",
    title = "ROC")
    for arg in args
        plot!(arg[1], arg[2], label = arg[3], lw = 2)
    end
    return p
end

plargs = [(fprvec, recvec, "VAE")]
display(plotroc(plargs...))
if !isinteractive()
    gui()
end

using MLBase: false_positive_rate, false_negative_rate
n = 21
alphavec = linspace(0,1,n)
eervec = zeros(n)
for i in 1:n
    model.alpha = alphavec[i]
    AnomalyDetection.setthreshold!(model, X)
    tryhat, tsthat, trroc, tstroc = AnomalyDetection.rocstats(dataset.data, dataset.labels,
        dataset.data, dataset.labels, model, verb = false)
    eervec[i] = (false_positive_rate(tstroc) + false_negative_rate(tstroc))/2
end

plot(alphavec, eervec, title = "equal error rate vs alpha",
    xlabel="alpha", ylabel = "EER")
if !isinteractive()
    gui()
end
