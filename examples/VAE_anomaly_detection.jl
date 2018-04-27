
using Plots
plotly()
import Plots: plot
clibrary(:Plots)
using JLD
code_path = "../src/"
push!(LOAD_PATH, code_path)
using AnomalyDetection

# load data
dataset = load("toy_data_3.jld")["data"]
X = AnomalyDetection.Float.(dataset.data)
Y = dataset.labels
nX = X[:, Y.==0]

# VAE settings
indim = size(X,1)
hiddendim = 10
latentdim = 2
nlayers = 2

# setup the VAE object
esize = [indim; hiddendim; hiddendim; latentdim*2] # encoder architecture
dsize = [latentdim; hiddendim; hiddendim; indim] # decoder architecture
lambda = 0.01 # KLD weight in loss function
threshold = 0 # classification threshold, is recomputed using setthreshold!()
contamination = size(Y[Y.==1],1)/size(Y, 1) # for automatic threshold computation
iterations = 2000
cbit = 500 # after this number of iteratiosn, callback is printed
verbfit = true
M = 1 # reconstruction error samples, for training 1 is OK
L = 50 # batchsize
# set low for training but high for classification
activation = Flux.relu
layer = Flux.Dense
rdelta = 1e-3 # reconstruction error threshold for training stopping
Beta = 1.0 # for automatic threshold computation, in [0, 1]
# 1.0 = tight around normal samples
tracked = true # do you want to store training progress?
# it can be later retrieved from model.traindata
astype = "logpn"
model = VAEmodel(esize, dsize, lambda, threshold, contamination, iterations, cbit, verbfit,
    L, M=M, activation = activation, layer = layer, rdelta = rdelta, Beta = Beta,
    tracked = tracked, astype = astype)

# fit the model
AnomalyDetection.evalloss(model, nX)
AnomalyDetection.fit!(model, nX)
AnomalyDetection.evalloss(model, nX)
AnomalyDetection.setthreshold!(model, X)

"""
	plot(model)

Plot the model loss.
"""
function plot(model::VAEmodel)
	# plot model loss
	if model.history == nothing
		println("No data to plot, set tracked = true before training.")
		return
	else
        p = plot(model.history[:loss], title = "model loss", label = "loss",
            xlabel = "iteration", ylabel = "loss + reconstruction error",
            seriestype = :line,
            markershape = :none)
        plot!(model.history[:reconstruction_error], label = "reconstruction error",
            seriestype = :line, markershape = :none, title = "model loss")
        plot!(model.history[:KLD], label = "KLD",
            seriestype = :line, markershape = :none,
            c = :green,
            title = "model loss")
        return p
    end
end


display(plot(model))
if !isinteractive()
    gui()
end

model(nX)

nX

AnomalyDetection.mu(model, nX)

AnomalyDetection.sigma(model, nX)

AnomalyDetection.sample_z(model, nX)

# predict labels
model.M = 100 # for classification higher is better (more stable)
tryhat = AnomalyDetection.predict(model, X)

# get the labels and roc stats
tryhat, tstyhat = AnomalyDetection.rocstats(dataset, dataset, model);

# plot heatmap of the fit
xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)
p = scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = :red, label = "predicted positive",
    xlims=xl, ylims = yl, title = "classification results")
scatter!(X[1, tryhat.==0], X[2, tryhat.==0], c = :green, label = "predicted negative",
    legend = (0.7, 0.7))

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


p = scatter(z1[1,:], z1[2,:], label = "first cluster", title = "code distribution")
scatter!(z2[1,:], z2[2,:], label = "second cluster")
scatter!(z3[1,:], z3[2,:], label = "third cluster")
scatter!(za[1,:], za[2,:], markersize = 3, label = "anomalous data")

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
