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
        x = Array{Float,2}(undef,0,0)
    end
    return x
end

""" 
    Basicset(path)

Outer constructor for the Basicset struct using a folder in the Loda database.
Transposes the arrays so that instances are columns.
"""
Basicset(path::String) = (isdir(path)) ? Basicset(
    txt2array(joinpath(path, "normal.txt"))',
    txt2array(joinpath(path, "easy.txt"))',
    txt2array(joinpath(path, "medium.txt"))',
    txt2array(joinpath(path, "hard.txt"))',
    txt2array(joinpath(path, "very_hard.txt"))',
    ) : error("No such path exists.")

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
function normalize(Y::Array{T,2} where T<:Real)
    M, N = size(Y)
    mu = mean(Y,dims=2);
    sigma = var(Y,dims=2);

    # if there are NaN present, then sigma is zero for a given column -> 
    # the scaled down column is also zero
    # but we treat this more economically by setting the denominator for a given column to one
    # also, we deal with numerical zeroes
    den = sigma
    den[abs.(den) .<= 1e-15] .= 1.0
    den[den .== 0.0] .= 1.0
    den = repeat(sqrt.(den), 1, N)
    nom = Y - repeat(mu, 1, N)
    nom[abs.(nom) .<= 1e-8] .= 0.0
    Y = nom./den
    return Y
end

"""
   normalize(x,y)

Concatenate x and y along the 2nd axis, normalize them and split them again. 
"""
function normalize(x,y)
    M,N = size(x)
    data = cat(x, y, dims = 2)
    data = normalize(data)
    return data[:, 1:N], data[:, N+1:end]
end

"""
    test01(x, msg)

Test if x in [0,1], else throw msg error.
"""
function test01(x::Real, msg)
    if !(0 <= x <= 1)
        error(msg)
    end
end

"""
    splitdata(x, alpha)

Split x to parts according to alpha.
"""
function splitdata(x, alpha)
    # test correct parameters size
    test01(alpha, "alpha must be in the interval [0,1]")

    M, N = size(x)
    trN = Int(floor(N*alpha))
    inds = sample(1:N, N, replace = false)
    trdata = x[:, inds[1:trN]]
    tstdata = x[:, inds[trN+1:end]]
    return trdata, tstdata
end

"""
    clusterdness(normal, anomalous)

compute the clusterdness - sample variance of normal vs anomalous instances.
"""
function clusterdness(normal, anomalous)
    M, N = size(normal)
    M, K = size(anomalous)

    varN = mean(pairwise(Euclidean(), normal[:, sample(1:N, min(1000, N), replace=false)]))/2
    varA = mean(pairwise(Euclidean(), anomalous[:, sample(1:K, min(1000, K), replace=false)]))/2

    (varA>0) ? (return varN/varA) : (return Inf)
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
function makeset(dataset::Basicset, alpha::Real=0.8, 
    difficulty::Union{String, Array{String,1}}="", 
    frequency::Real=0.05, variation::String="low"; seed=false)
    # test correct parameters size
    test01(frequency, "frequency must be in the interval [0,1]")

    # first extract the basic normal and anomalous data
    normal = dataset.normal

    # problem dimensions
    M, N = size(normal)
    trN = Int(floor(N*alpha))
    tstN = N-trN

    # get all anomalies
    if difficulty == ""
        anomalous = Array{Float,2}(M,0)
        for dif in intersect([:easy, :medium, :hard, :very_hard], fieldnames(dataset))
            _X = getfield(dataset, dif)
            if length(_X) > 0
                anomalous = cat(2, anomalous, _X)
            end
        end   
    # select only some anomaly types
    elseif typeof(difficulty) == Array{String,1}
        anomalous = Array{Float,2}(undef, M,0)
        for dif in intersect([:easy, :medium, :hard, :very_hard], Symbol.(difficulty), 
            fieldnames(typeof(dataset)))
            _X = getfield(dataset, dif)
            if length(_X) > 0
                anomalous = cat(anomalous, _X, dims = 2)
            end
        end
    # get just one type
    else
        anomalous = getfield(dataset, parse(difficulty))
    end

    # check if any anomalies are actually sampled from
    if length(anomalous)==0
        error("no anomalous data of given difficulty level")
    end

    # set seed
    if seed != false
        Random.seed!(seed)
    end


    # normalize the data to zero mean and unit variance    
    normal, anomalous = normalize(normal, anomalous)

    # randomly sample the training and testing normal data
    trNdata, tstNdata = splitdata(normal, alpha)

    if difficulty == ""
        trAdata, tstAdata = splitdata(anomalous, alpha)
        tstK = size(tstAdata,2)
        trK = size(trAdata,2)
    # select only some anomaly types
    elseif typeof(difficulty) == Array{String,1}
        aM, aN = size(anomalous)
        # shuffle data
        anomalous = anomalous[:,sample(1:aN, aN, replace = false)]
        # how many anomalies in the training dataset
        trK = min(Int(round(trN*frequency)), Int(round(aN*alpha)))
        trAdata = anomalous[:,1:trK]

        # put the rest in teh testing dataset
        tstAdata = anomalous[:,trK+1:end]
        tstK = size(tstAdata,2)
    # this is done when we don't want to select all anomalies
    else
        # how many anomalous points to be sampled 
        aM, aN = size(anomalous)
        trK = min(Int(round(trN*frequency)), Int(round(aN*alpha)))
        tstK = min(Int(round(tstN*frequency)), Int(round(aN*(1-alpha)))) 


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
    end

    # restart the seed
    Random.seed!()
    
    c = clusterdness(normal, anomalous)

    # finally, generate the dataset
    trData = Dataset(
        cat(trNdata, trAdata, dims = 2),
        cat(zeros(trN), ones(trK), dims = 1)
        )

    tstData = Dataset(
        cat(tstNdata, tstAdata, dims = 2),
        cat(zeros(tstN), ones(tstK), dims = 1)
        )

    return trData, tstData, c
end

"""
    getdata(datasetname, alpha=0.8, difficulty="", frequency=0.05, 
            variation="low"; seed=false)

Returns a training and testing Dataset. If difficulty = "" (default),
all anomalies are sampled from.
"""
function getdata(datasetname::String, alpha::Real=0.8, 
    difficulty::Union{String, Array{String,1}}="", 
    frequency::Real=0.05, variation::String="low";seed=false, loc="")
    if loc == ""
        datapath = datasetpath()
    else
        datapath = loc
    end
    bs = Basicset(joinpath(datapath,datasetname))
    return makeset(bs,alpha,difficulty,frequency,variation,seed=seed)
end

"""
    datasetpath()

Return absolute path of benchmark datasets.
"""
datasetpath() =
    joinpath(joinpath(dirname(pathof(AnomalyDetection)), ".."),"experiments/datasets")

"""
    datasetnames()

Return list of names of available datasets.
"""
datasetnames() = readdir(datasetpath())

"""
    getrawdata(dataset)

Returns a given Basicset.
"""
getrawdata(dataset::String) = Basicset(joinpath(datasetpath(),dataset))

"""
    cat(bs::Basicset)

Return an array consisting of all concatenated arrays in bs and 
indices identifying the original array boundaries.
"""
function cat(bs::Basicset)
    X = bs.normal
    inds = [size(X,2)]
    for field in filter(x -> x != :normal, [f for f in fieldnames(typeof(bs))])
        x = getfield(bs,field)
        m = size(x,2)
        if m!= 0
            X = cat(X,x,dims=2)
        end
        push!(inds, m)
    end
    return X, inds
end

"""
    getthreshold(model, X, contamination, [asargs; beta, askwargs])

Compute threshold for model classification based on known contamination level.
"""
function getthreshold(model, x, contamination, asargs...; Beta = 1.0, askwargs...)
    N = size(x, 2)
    Beta = Float(Beta)
    # get anomaly score
    ascore  = anomalyscore(model, x, asargs...; askwargs...)
    # sort it
    ascore = sort(ascore)
    aN = Int(ceil(N*contamination)) # number of contaminated samples
    # get the threshold - could this be done more efficiently?
    (aN > 0) ? (return Beta*ascore[end-aN] + (1-Beta)*ascore[end-aN+1]) : (return ascore[end])
end

"""
    labels2bin(y)

Changes binary coded array from {-1,1} to {0,1}.
"""
function labels2bin(y::Array{Int64,1})
    x = copy(y)
    x[x.==-1] .= 0
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

#####################################
#### auxiliary and loss functions ###
#####################################

""" 
    softplus(X)

softplus(X) = log(exp(X) + 1)   
"""
softplus(X) = log.(exp.(X) .+ 1)

"""
    KL(μ, σ2)

KL divergence between a normal distribution and unit gaussian.
"""
KL(μ, σ2) = Float(1/2)*mean(sum(σ2 + μ.^2 - log.(σ2) .- 1, dims = 1))

"""
    likelihood(X, μ, [σ2])

Likelihood of a sample X given mean and variance.
"""
likelihood(X, μ) = - mean(sum((μ - X).^2,dims = 1))/2
likelihood(X, μ, σ2) = - mean(sum((μ - X).^2 ./σ2 + log.(σ2),dims = 1))/2

"""
    mu(X)

Extract mean as the first horizontal half of X.
"""
mu(X) = X[1:Int(size(X,1)/2),:]

"""
    sigma2(X)

Extract sigma^2 as the second horizontal half of X. 
"""
sigma2(X) = softplus(X[Int(size(X,1)/2+1):end,:]) .+ Float(1e-6)

"""
    logps(x)

Is the logarithm of the standard pdf of x.
"""
logps(x) = abs.(-1/2*x.^2 - 1/2*log(2*pi))

"""
    samplenormal(X)

Sample normal distribution with mean and sigma2 extracted from X.
"""
function samplenormal(X)
    μ, σ2 = mu(X), sigma2(X)
    ϵ = Float.(randn(size(μ)))
    return μ .+  ϵ .* sqrt.(σ2)
end

"""
    k(x,y,σ)

Gaussian kernel of x and y.
"""
k(x,y,σ) = exp.(-(sum((x-y).^2,dims=1)/(2*σ)))

"""
    ekxy(X,Y,σ)

E_{x in X,y in Y}[k(x,y,σ)] - mean value of kernel k.
"""
ekxy(X,Y,σ) = mean(k(x,y,σ))

"""
    MMD(X,qz_sampler,pz_sampler,σ)

MMD of qz and pz given data matrix X.    
"""
MMD(X,qz_sampler,pz_sampler,σ) = ekxy(qz_sampler(X),qz_sampler(X),σ) - 
    2*ekxy(qz_sampler(X), pz_sampler(X),σ) + 
        ekxy(pz_sampler(X),pz_sampler(X),σ)

#####################################
#####################################

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