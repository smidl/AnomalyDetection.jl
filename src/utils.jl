"""
Structure representing the basic Loda anomaly dataset.
"""
struct Basicset
    normal::Array{Float, 2}
    easy::Array{Float, 2}
    medium::Array{Float, 2}
    hard::Array{Float, 2}
    very_hard::Array{Float, 2}
end

"""
Structure representing a dataset.
"""
mutable struct Dataset
    data::Array{Float,2}
    labels::Array{Int64,1}
end

"""
   txt2array(file)

If the file does not exist, returns an empty 2D array. 
"""
function txt2array(file::String)
    if isfile(file)
        x = readdlm(file)
    else
        x = Array{Float,2}(0,0)
    end
    return x
end

""" 
    Basicset(path)

Outer constructor for the Basicset struct using a folder in the Loda database.
Transposes the arrays so that instances are columns.
"""
Basicset(path::String) = Basicset(
    txt2array(joinpath(path, "normal.txt"))',
    txt2array(joinpath(path, "easy.txt"))',
    txt2array(joinpath(path, "medium.txt"))',
    txt2array(joinpath(path, "hard.txt"))',
    txt2array(joinpath(path, "very_hard.txt"))',
    )

"""
    loaddata(masterpath)

Loads all the data from the Loda database.
"""
function loaddata(masterpath::String)
    data = Dict{String, Basicset}()
    contents = readdir(masterpath)
    for name in contents
        path = joinpath(masterpath, name)
        if isdir(path)
            data[name] = Basicset(path)
        end
    end

    return data
end

"""
    normalize(Y)

Scales down a 2 dimensional array so it has approx. standard normal distribution. 
Instance = column. 
"""
function normalize(Y::Array{Float,2})
    M, N = size(Y)
    mu = mean(Y,2);
    sigma = var(Y,2);

    # if there are NaN present, then sigma is zero for a given column -> 
    # the scaled down column is also zero
    # but we treat this more economically by setting the denominator for a given column to one
    # also, we deal with numerical zeroes
    den = sigma
    den[abs.(den) .<= 1e-15] = 1.0
    den[den .== 0.0] = 1.0
    den = repmat(sqrt.(den), 1, N)
    nom = Y - repmat(mu, 1, N)
    nom[abs.(nom) .<= 1e-8] = 0.0
    Y = nom./den
    return Y
end

"""
    trData, tstData, clusterdness = makeset(dataset, alpha, difficulty, frequency, variation, [normalize, seed])

Sample a given dataset, return training and testing subsets and a measure of clusterdness. 
See Emmott, Andrew F., et al. "Systematic construction of anomaly detection benchmarks from 
real data.", 2013 for details.

alpha - the ratio of training to all data\n
difficulty - easy/medium/hard/very_hard problem based on similarity of anomalous measurements to normal\n
frequency - ratio of anomalous to normal data\n
variation - low/high - should anomalies be clustered or not\n
seed - random seed
"""
function makeset(dataset::Basicset, alpha::Real, difficulty::String, frequency::Real, variation::String;
                 seed=false)
    # first extract the basic normal and anomalous data
    normal = dataset.normal
    anomalous = getfield(dataset, parse(difficulty))
    if length(anomalous)==0
        error("no anomalous data of given difficulty level")
    end

    # test correct parameters size
    if !(0 <= alpha <= 1)
        error("alpha must be in the interval [0,1]")
    end
    if !(0 <= frequency <= 1)
        error("frequency must be in the interval [0,1]")
    end

    # problem dimensions
    M, N = size(normal)
    trN = Int(floor(N*alpha))
    tstN = N-trN

    # how many anomalous points to be sampled 
    aM, aN = size(anomalous)
    trK = min(Int(round(trN*frequency)), Int(round(aN*alpha)))
    tstK = min(Int(round(tstN*frequency)), Int(round(aN*(1-alpha)))) 

    # set seed
    if seed != false
        srand(seed)
    end

    # normalize the data to zero mean and unit variance    
    data = cat(2, normal, anomalous)
    data = normalize(data)
    normal = data[:, 1:N]
    anomalous = data[:, N+1:end]

    # randomly sample the training and testing normal data
    inds = sample(1:N, N, replace = false)
    trNdata = normal[:, inds[1:trN]]
    tstNdata = normal[:, inds[trN+1:end]]

    # now sample the anomalous data
    K = trK + tstK
    #if K > aN
        #error("not enough anomalous data to sample from")
        #warning("not enough anomalous data to sample from")
    #end
    if variation == "low"
        # in this setting, simply sample trK and tstK anomalous points
        # is this done differently in the original paper?
        inds = sample(1:aN, K, replace = false)
    elseif variation == "high"
        # in this setting, randomly choose a point and then K-1 nearest points to it as a cluster
        ind = sample(1:aN, 1)
        x = anomalous[:, ind]
        x = reshape(x, length(x), 1) # promote the vector to a 2D array
        # here maybe other metrics could be used?
        dists = pairwise(Euclidean(), x, anomalous) # get the distance vector
        inds = sortperm(reshape(dists, length(dists))) # get the sorted indices
        inds = inds[1:K] # get the nearest ones
        inds = inds[sample(1:K, K, replace=false)] # scramble them
    end
    anomalous = anomalous[:, inds]
    trAdata = anomalous[:,1:trK]
    tstAdata = anomalous[:,trK+1:end]

    # compute the clusterdness - sample variance of normal vs anomalous instances
    varN = mean(pairwise(Euclidean(), normal[:, sample(1:N, min(1000, N), replace=false)]))/2
    varA = mean(pairwise(Euclidean(), anomalous[:, sample(1:K, min(1000, K), replace=false)]))/2

    if varA>0
        clusterdness = varN/varA
    else
        clusterdness = Inf
    end

    # finally, generate the dataset
    trData = Dataset(
        cat(2, trNdata, trAdata),
        cat(1, zeros(trN), ones(trK))
        )

    tstData = Dataset(
        cat(2, tstNdata, tstAdata),
        cat(1, zeros(tstN), ones(tstK))
        )

    return trData, tstData, clusterdness
end

"""
    labels2bin(y)

Changes binary coded array from {-1,1} to {0,1}.
"""
function labels2bin(y::Array{Int64,1})
    x = copy(y)
    x[x.==-1] = 0
    return x;
end

"""
    bin2labels(y)

Changes binary coded array from {0,1} to {-1,1}.
"""
function bin2labels(y::Array{Int64,1})
    x = copy(y)
    x[x.==0] = -1
    return x;
end

"""
    switchlabels(y)

Swaps labels in a binary vector of {0,1}.
"""
function switchlabels(y::Array{Int64,1})
    x = copy(y);
    x[y.==0] = 1;
    x[y.==1] = 0;
    return x;
end

"""
    quickvalidate(data, samplesettings, algorithm, [supervised, verb])

Quickly validate an algorithm on a dataset.
ScikitLearn version (instances in rows) with data sampling from BasicSet.
"""
function quickvalidate!(data::Basicset, settings::Dict{Any,Any}, algorithm; 
    supervised::Bool = false, verb = true)
    # sample the data
    trData, tstData, c = makeset(data, settings["alpha"], settings["difficulty"],
                                    settings["frequency"], settings["variation"])

    print("clusterdness = $(c)\n")

    return quickvalidate!(trData, tstData, algorithm, supervised = supervised, verb = verb)
end

"""
    quickvalidate(trData, tstData, algorithm, [supervised, verb])

Quickly validate an algorithm on a dataset.
ScikitLearn version (instances in rows) with sampled data.
"""
function quickvalidate!(trData::Dataset, tstData::Dataset, algorithm; 
    supervised::Bool = false, verb = true)
    # fit the algorithm with the training data
    if supervised
        fit!(algorithm, trData.data', trData.labels)
    else
        fit!(algorithm, trData.data')
    end

    return rocstats(trData.data', trData.labels, tstData.data', tstData.labels,
        algorithm, verb = verb)
end

"""
    quickvalidate(trData, tstData, algorithm, [verb])

Quickly validate an algorithm on a dataset.
VAE version (instances in columns) with known contamination level.
"""
function quickvalidate!(trData::Dataset, tstData::Dataset, algorithm::VAEmodel; verb = true)
    # fit the model
    # only non-anomalous data are used for training
    fit!(algorithm, trData.data, trData.labels)

    return rocstats(trData.data, trData.labels, tstData.data, tstData.labels, algorithm, verb = verb)
end

"""
    quickvalidate(trData, tstData, algorithm, [verb])

Quickly validate an algorithm on a dataset.
AE version (instances in columns) with known contamination level.
"""
function quickvalidate!(trData::Dataset, tstData::Dataset, algorithm::AEmodel; verb = true)
    # fit the model
    # only non-anomalous data are used for training
    fit!(algorithm, trData.data, trData.labels)

    return rocstats(trData.data, trData.labels, tstData.data, tstData.labels, algorithm, verb = verb)
end

"""
    quickvalidate(trData, tstData, algorithm, [verb])

Quickly validate an algorithm on a dataset.
GAN version (instances in columns) with known contamination level.
"""
function quickvalidate!(trData::Dataset, tstData::Dataset, algorithm::GANmodel; verb = true)
    # fit the model
    # only non-anomalous data are used for training
    fit!(algorithm, trData.data, trData.labels)

    return rocstats(trData.data, trData.labels, tstData.data, tstData.labels, algorithm, verb = verb)
end

"""
    rocstats(trX, trY, tstX, tstY, algorithm, [verb])

Quickly validate an algorithm on a dataset, an instance is a column of X.
"""
function rocstats(trX, trY, tstX, tstY, algorithm; verb = true)
    # get the results on the training dataset
    tryhat = correctlabels(algorithm, trX, trY)

    # measures of accuracy
    trroc = roc(trY, tryhat);
    if verb
        print("\n Training data performance: \n")
        print(trroc)
        print("precision: $(precision(trroc))\n")
        print("f1score: $(f1score(trroc))\n")
        print("recall: $(recall(trroc))\n")
        print("false positive rate: $(false_positive_rate(trroc))\n")
        print("equal error rate: $((false_positive_rate(trroc) + false_negative_rate(trroc))/2)\n")
    end

    # accuracy on test data
    tstyhat = correctlabels(algorithm, tstX, tstY)
    
    # measures of accuracy
    tstroc = roc(tstY, tstyhat);
    if verb
        print("\n Testing data performance: \n")
        print(tstroc)
        print("precision: $(precision(tstroc))\n")
        print("f1score: $(f1score(tstroc))\n")
        print("recall: $(recall(tstroc))\n")
        print("false positive rate: $(false_positive_rate(tstroc))\n")
        print("equal error rate: $((false_positive_rate(tstroc) + false_negative_rate(tstroc))/2)\n")
    end

    return tryhat, tstyhat, trroc, tstroc
end

"""
    correctlabels(algorithm, X, Y)

Predicts binary labels of X and switches them of needed in order to 
respect correct label order. 
"""
function correctlabels(algorithm, X, Y)
    # compute labels prediction
    Yhat = labels2bin(predict(algorithm, X));
    ROC = roc(Y, Yhat);
    # the labels may be switched if the ordering is incorrect
    if recall(ROC) < false_positive_rate(ROC)
        Yhat = switchlabels(Yhat);
    end
    return Yhat

end

"""
    rocstats(trdata, tstdata, algorithm, [verb])

Return labels and ROC statistics.
"""
rocstats(trdata::Dataset, tstdata::Dataset, algorithm; verb = true) = 
 rocstats(trdata.data, trdata.labels, tstdata.data, tstdata.labels, algorithm; verb = true)

"""
    getroccurve(ascorevec, labels)

Returns data for drawing the roc curve
"""
function getroccurve(ascorevec, labels)
    N = size(labels,1)
    @assert N == size(ascorevec,1)
    if isnan(ascorevec[1])
        warn("Anomaly score is NaN, check your inputs!")
    end
    
    fprvec = Array{AnomalyDetection.Float,1}(N+2)
    recvec = Array{AnomalyDetection.Float,1}(N+2)
    p = sum(labels)
    n = N - p
    fpr = 1.0
    rec = 1.0
    fprvec[1] = fpr # fp/n
    recvec[1] = rec # tp/p
    sortidx = sortperm(ascorevec)
    for i in 2:(N+1)
        (labels[sortidx[i-1]] == 0)? (fpr = fpr - 1/n) : (rec = rec -1/p)
        if (fpr <= rec)
            fprvec[i] = max(0,fpr)
            recvec[i] = max(0,rec)
        else
            fprvec[i] = max(0,1-fpr)
            recvec[i] = max(0,1-rec)
        end
    end
    
    # ensure zeros
    recvec[end] = 0.0
    fprvec[end] = 0.0
    
    # sort them
    isort = sortperm(fprvec)
    recvec = recvec[isort]
    fprvec = fprvec[isort]
    
    # avoid regression
    for i in 2:(N+2)
        if recvec[i] < recvec[i-1]
            recvec[i] = recvec[i-1]
        end
    end
    
    return recvec, fprvec
end

"""
    auc(x,y, [weights])

Computes the are under curve (x,y).
"""
function auc(x,y, weights = "same")
    # compute the increments
    dx = x[2:end] - x[1:end-1]
    dy = y[2:end] - y[1:end-1]

    a = y[1:end-1] + dy/2

    if weights == "same"
        a = y[1:end-1] + dy/2
        b = dx
    elseif weights == "1/x"
        inz = x.!=0 # nonzero indices
        w = 1./x[inz]
        # w = w/sum(w) # this is numerically unstable
        a = (y[1:end-1] + dy/2)[inz[2:end]]
        a = a.*w
        b = dx[inz[2:end]]
    end
        
    return dot(a,b)
end

"""
    mprint(string, [verb])

Muted print: if verb = true (default), print the string.    
"""
function mprint(string; verb = true)
    if verb
        print(string)
    end
end

"""
    mprintln(string, [verb])

Muted println: if verb = true (default), println the string.    
"""
function mprintln(string; verb = true)
    if verb
        println(string)
    end
end

""" 
    softplus(X)

softplus(X) = log(exp(X) + 1)   
"""
softplus(X) = log.(exp.(X)+1)

"""
    freeze(m)

Creates a non-trainable copy of a Flux object.
"""
freeze(m) = Flux.mapleaves(Flux.Tracker.data,m)

#"""
#    adapt(T, array)
#
#Convert array to type T.
#"""
#adapt(T, x::Array) = T.(x)

#"""
#    adapt(T, Flux.Dense)
#
#Convert params of Dense layer to type T.
#"""
#adapt(T, m::Flux.Dense) = Flux.Dense(adapt(T,m.W),adapt(T,m.b),m.σ)

"""
    adapt(T, Flux.Chain)

Convert params of a whole chain to type of T.
"""
adapt(T, m::Flux.Chain) = Flux.Chain(map(l -> FluxExtensions.adapt(T,l), m.layers)...)

"""
    aelayerbuilder(lsize, activation, layer)

Construct encoder/decoder using FluxExtensions.
"""
aelayerbuilder(lsize::Vector, activation, layer) = adapt(Float, 
    FluxExtensions.layerbuilder(lsize, 
    Array{Any}(fill(layer, size(lsize,1)-1)), 
    Array{Any}([fill(activation, size(lsize,1)-2); identity]))
    )

"""
    discriminatorbuilder(lsize, activation, layer)

Construct discriminator with last sigmoid output.
"""
discriminatorbuilder(lsize::Vector, activation, layer) = adapt(Float, 
    FluxExtensions.layerbuilder(lsize, 
    Array{Any}(fill(layer, size(lsize,1)-1)), 
    Array{Any}([fill(activation, size(lsize,1)-2); σ]))
    )