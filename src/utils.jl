import Base.convert
using MLBase: roc, correctrate, precision, recall, f1score, false_positive_rate, false_negative_rate
#using ScikitLearn: @sk_import, fit!, predict
using StatsBase: sample


"""
Structure representing the basic Loda anomaly dataset.
"""
struct Basicset
    normal::Array{Float64, 2}
    easy::Array{Float64, 2}
    medium::Array{Float64, 2}
    hard::Array{Float64, 2}
    very_hard::Array{Float64, 2}
end

"""
Structure representing a dataset.
"""
mutable struct Dataset
    data::Array{Float64,2}
    labels::Array{Int64,1}
end

"""
   txt2array(file::String)

If the file does not exist, returns an empty 2D array. 
"""
function txt2array(file::String)
    if isfile(file)
        x = readdlm(file)
    else
        x = Array{Float64,2}(0,0)
    end
    return x
end

""" 
    Basicset(path::String)

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
    loaddata(masterpath::String)

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
    normalize(Y::Array{Float64,2})

Scales down a 2 dimensional array so it has approx. standard normal distribution. 
Instance = column. 
"""
function normalize(Y::Array{Float64,2})
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
    normalize(Y::Array{Float32,2})

Scales down a 2 dimensional array so it has approx. standard normal distribution.
"""
function normalize(Y::Array{Float32,2})
    Y64 = convert(Array{Float64,2}, Y)
    Y64 = scaley(Y64)
    return convert(Array{Float32,2}, Y64)
end

"""
    (dataset::Basicset, alpha::Float64, difficulty::String, frequency::Float64, variation::String; 
    normalize=true, seed=false)

Sample a given dataset, return training and testing subsets and a measure of clusterdness. 
See Emmott, Andrew F., et al. "Systematic construction of anomaly detection benchmarks from 
real data.", 2013 for details.

alpha - the ratio of training to all data\n
difficulty - easy/medium/hard/very_hard problem based on similarity of anomalous measurements to normal\n
frequency - ratio of anomalous to normal data\n
variation - low/high - setting of whether data should be clustered or not\n
seed - random seed
"""
function makeset(dataset::Basicset, alpha::Float64, difficulty::String, frequency::Float64, variation::String;
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
    trK = Int(ceil(trN*frequency)) 
    tstK = Int(ceil(tstN*frequency)) 

    # set seed
    if seed
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
    if K > aN
        error("not enough anomalous data to sample from")
    end
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
    labels2bin(y::Array{Int64,1})

Changes binary coded array from {-1,1} to {0,1}.
"""
function labels2bin(y::Array{Int64,1})
    x = copy(y)
    x[x.==-1] = 0
    return x;
end

"""
    bin2labels(y::Array{Int64,1})

Changes binary coded array from {0,1} to {-1,1}.
"""
function bin2labels(y::Array{Int64,1})
    x = copy(y)
    x[x.==0] = -1
    return x;
end

"""
    switchlabels(y::Array{Int64,1})

Swaps labels in a binary vector of {0,1}.
"""
function switchlabels(y::Array{Int64,1})
    x = copy(y);
    x[y.==0] = 1;
    x[y.==1] = 0;
    return x;
end

"""
    quickvalidate(data::BasicSet, samplesettings::Dict{Any,Any}, algorithm; 
    supervised::Bool = false)

Quickly validate an algorithm on a dataset.
ScikitLearn version (instances in rows) with data sampling from BasicSet.
"""
function quickvalidate!(data::Basicset, settings::Dict{Any,Any}, algorithm; 
    supervised::Bool = false)
    # sample the data
    trData, tstData, c = makeset(data, settings["alpha"], settings["difficulty"],
                                    settings["frequency"], settings["variation"])

    print("clusterdness = $(c)\n")

    return quickvalidate!(trData, tstData, algorithm, supervised = supervised)
end

"""
    quickvalidate(trData::Dataset, ststData::Dataset, algorithm; 
    supervised::Bool = false)

Quickly validate an algorithm on a dataset.
ScikitLearn version (instances in rows) with sampled data.
"""
function quickvalidate!(trData::Dataset, tstData::Dataset, algorithm; 
    supervised::Bool = false)
    # fit the algorithm with the training data
    if supervised
        fit!(algorithm, trData.data', trData.labels)
    else
        fit!(algorithm, trData.data')
    end

    return rocstats(trData.data', trData.labels, tstData.data', tstData.labels,
        algorithm)
end

"""
    quickvalidate(trData::Dataset, ststData::Dataset, algorithm, contamination)

Quickly validate an algorithm on a dataset.
VAE version (instances in columns) with known contamination level.
"""
function quickvalidate!(trData::Dataset, tstData::Dataset, algorithm::VAEmodel)
    # fit the model
    # only non-anomalous data are used for training
    fit!(algorithm, trData.data[:,trData.labels .== 0])

    return rocstats(trData.data, trData.labels, tstData.data, tstData.labels, algorithm)
end

"""
    rocstats(trX, trY, tstX, tstY, algorithm)

Quickly validate an algorithm on a dataset, an instance is a column of X.
"""
function rocstats(trX, trY, tstX, tstY, algorithm)
    # get the results on the training dataset
    tryhat = labels2bin(predict(algorithm, trX));
    cr = correctrate(trY, tryhat);
    # the labels may be switched
    if cr < 0.5
        tryhat = switchlabels(tryhat);
    end

    # measures of accuracy
    print("\n Training data performance: \n")
    trroc = roc(trY, tryhat);
    print(trroc)
    print("precision: $(precision(trroc))\n")
    print("recall: $(recall(trroc))\n")
    print("f1score: $(f1score(trroc))\n")
    print("equal error rate: $((false_positive_rate(trroc) + false_negative_rate(trroc))/2)\n")

    # accuracy on test data
    tstyhat = labels2bin(predict(algorithm, tstX));
    cr = correctrate(tstY, tstyhat);
    # the labels may be switched
    if cr < 0.5
        tstyhat = switchlabels(tstyhat);
    end

    # measures of accuracy
    print("\n Testing data performance: \n")
    tstroc = roc(tstY, tstyhat);
    print(tstroc)
    print("precision: $(precision(tstroc))\n")
    print("recall: $(recall(tstroc))\n")
    print("f1score: $(f1score(tstroc))\n")
    print("equal error rate: $((false_positive_rate(tstroc) + false_negative_rate(tstroc))/2)\n")

    return tryhat, tstyhat
end

