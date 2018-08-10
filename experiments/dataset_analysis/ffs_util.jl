using AnomalyDetection,EvalCurves, DataFrames
include("../eval.jl")
using Combinatorics, ProgressMeter

function idxpairs(N)
    ps = []
    for x in combinations(1:N,2)
        push!(ps,x)
    end
    return ps 
end

scramble(x) = x[sample(1:length(x),length(x),replace=false)]
subfeatures(d::AnomalyDetection.Dataset, inds) = Dataset(d.data[inds,:], d.labels)
subfeatures(data,inds) = (subfeatures(data[1], inds),subfeatures(data[2], inds)) 


function knnscoreall(trdata, tstdata)
    kvec = [1,3,5,11,27]
    aucvec = []
    for k in kvec
        auc, _, _ = knnscore(trdata,tstdata,k)
        push!(aucvec, auc)
    end
    mx = findmax(aucvec)
    return mx[1], kvec[mx[2]]
end

function knnscore(trdata,tstdata,k)
    # only use nonanomalous data for training
    trx = trdata.data[:,trdata.labels.==0]
    # construct and fit the model
    model = AnomalyDetection.kNN(k, 0.1)
    AnomalyDetection.fit!(model, trx)
    # get auc on testing data
    as = AnomalyDetection.anomalyscore(model,tstdata.data)
    auc = EvalCurves.auc(EvalCurves.roccurve(as,tstdata.labels)...)
    return auc, model, as
end

function vaescore(trdata, tstdata)
    # only use nonanomalous data for training
    trx = trdata.data[:,trdata.labels.==0]
    M,N = size(trx)
    model = AnomalyDetection.VAE([M,4,8,4,2],[1,4,8,4,M])
    AnomalyDetection.fit!(model,trx,min(N,256),
        iterations = 10000,
        cbit=500, 
        lambda = 0.00001,
        verb = false
    )
    as = AnomalyDetection.anomalyscore(model,tstdata.data,1)
    auc = EvalCurves.auc(EvalCurves.roccurve(as,tstdata.labels)...)
    return auc, model, as
end

function getdata(dataset,alldata=true,seed=518;loc="")
    if alldata
        return AnomalyDetection.getdata(dataset, seed = seed, loc=loc)
    else
        if dataset in ["madelon", "gisette", "abalone", "haberman", "letter-recognition",
            "isolet", "multiple-features", "statlog-shuttle"]
            difficulty = "medium"
        elseif dataset in ["vertebral-column"]
            difficulty = "hard"
        else
            difficulty = "easy"
        end
        return AnomalyDetection.getdata(dataset, 0.8, difficulty, seed = seed, loc=loc)
    end
end

function scorefeatures(dataset, maxtries = 10, alldata = true)
    # dataframe to store results
    resdf = DataFrame(
        f1 = Int[],
        f2 = Int[],
        vae = Float64[],
        knn = Float64[],
        k = Int[]
        )
    
    # get all the data
    data = getdata(dataset, alldata)
    M,N = size(data[1].data)
    
    # create pairs
    ipairs = idxpairs(M)
    ipairs = scramble(ipairs)
    
    # progress
    imax = min(length(ipairs),maxtries)
    p = Progress(imax)
    
    for i in 1:imax
        pair = pop!(ipairs)
        trdata = subfeatures(data[1], pair)
        tstdata = subfeatures(data[2], pair)
        
        # get the kNN scores
        ks, k = knnscoreall(trdata,tstdata)
        # get VAE score
        vs,_,_ = vaescore(trdata,tstdata)
        
        push!(resdf, [pair[1], pair[2], vs, ks, k])
        next!(p)
    end
    return resdf
end
