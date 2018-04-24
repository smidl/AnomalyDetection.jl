using DataFrames, Query, FileIO, ValueHistories
import Missings: missing, skipmissing, ismissing

push!(LOAD_PATH, "../src")
using AnomalyDetection

function auroc(ascore, labels)
    if isnan(ascore[1])
        return missing
    else
        tprvec, fprvec = AnomalyDetection.getroccurve(ascore, labels)
        return AnomalyDetection.auc(fprvec, tprvec)
    end
end

function topprecision(ascore, labels, p)
    if isnan(ascore[1])
        return missing
    else
        N = size(ascore,1)
        @assert size(labels,1) == N
        topN = Int(round(N*p))
        si = 1:topN
        isort = sortperm(ascore, rev = true)
        sl = labels[isort][si]
        return sum(sl)/sum(labels[end-topN+1:end]) # precision = true positives/labeled positives
    end
end

function computestats(datapath, algnames)
    df = DataFrame()
    cnames = ["dataset", "algorithm", "iteration", "settings", "train_auroc", "test_auroc", 
        "top_5p"]
    for name in cnames
        df[Symbol(name)] = Any[]
    end
    
    datasets = readdir(datapath)
    for dataset in datasets
        path = joinpath(datapath, dataset)
        algs = readdir(path)
        for alg in intersect(algs, algnames)
            _path = joinpath(path, alg)
            iters = readdir(_path)
            for iter in iters
                __path = joinpath(_path, iter)
                ios = readdir(__path)
                for io in ios
                    f = joinpath(__path, io)
                    # compute training and testing auroc
                    trauroc = auroc(load(f, "training_anomaly_score"), load(f, "training_labels"))
                    tstauroc = auroc(load(f, "testing_anomaly_score"), load(f, "testing_labels"))
                    
                    # compute top 5% ascore samples precision
                    tp = topprecision(load(f, "training_anomaly_score"), load(f, "training_labels"), 0.05)
                    
                    # save the data
                    push!(df, [dataset, alg, iter, io, trauroc, tstauroc, tp])
                end
            end
        end
    end
    
    return df
end

# loads the complete df from csv and transfomrs strings to numbers
function loadtable(fname, datacols = 5:7)
    #load the df
    data = readtable(fname)
    
    for cname in names(data)
        data[cname] = Array{Any,1}(data[cname])
    end
    
    nrows, ncols = size(data)
    
    # go through the whole df and replace missing strings with actual Missing type
    # and floats with float
    for cname in names(data)[datacols]
        for i in 1:nrows
            (data[cname][i] == "missing")? data[cname][i]=missing : 
                data[cname][i]=float(data[cname][i])
        end
    end
    
    return data
end

function rankdf(df)
    _df = deepcopy(df)
    nrows, ncols = size(_df)
    nalgs = ncols - 1
    
    algnames = names(df)[2:end]
    
    for i in 1:nrows
        row = _df[i,2:end]
        arow = reshape(Array(row), nalgs)
        isort = sortperm(arow, rev = true)
        j = 1    
        tiec = 0 # tie counter
        # create ranks
        arow = collect(skipmissing(arow))
        for alg in algnames[isort]
            if ismissing(row[alg][1])
                _df[alg][i] = missing
            else
                # this decides ties
                val = row[alg][1]
                nties = size(arow[arow.==val],1) - 1
                if nties > 0
                    _df[alg][i] = (sum((j-tiec):(j+nties-tiec)))/(nties+1)
                    tiec +=1
                    # restart tie counter
                    if tiec > nties
                        tiec = 0
                    end
                else
                    _df[alg][i] = j
                end
                j+=1  
            end
        end
    end
    
    # append the final row with mean ranks
    push!(_df, cat(1,Array{Any}(["mean rank"]), zeros(nalgs)))
    for alg in algnames
        _df[alg][end] = mean(skipmissing(_df[alg][1:end-1]))
    end
    
    return _df
end

function missmean(x)
    if size(x,1) == 0
        return missing
    else
        return(mean(x))
    end
end

function missmax(x)
    if size(x,1) == 0
        return missing
    else
        return(maximum(x))
    end
end

function missfindmax(x)
   if size(x,1) == 0
        return missing
    else
        return(findmax(x))
    end
end 

function maxauroc(data, algs)
    df = DataFrame()
    df[:dataset] = String[]
    for alg in algs
        df[Symbol(alg)] = Any[]
    end
    nalgs = size(algs,1)
    
    # collect dataset names
    datasets = unique(data[:dataset])
    
    for (i,dataset) in zip(1:size(datasets,1), datasets)
        row = Array{Any,1}(nalgs+1)
        row[2:end] = missing
        row[1] = dataset
        push!(df, reshape(row, 1, nalgs+1))
        for alg in algnames
            dfx = @from r in data begin
                @where r.algorithm == alg && r.dataset == dataset
                @select {r.iteration, r.test_auroc}
                @collect DataFrame
            end
                
            try
                # group this by iterations
                dfx = by(dfx, [:iteration], 
                    d -> DataFrame(auroc = 
                        missmax(collect(skipmissing(d[:test_auroc])))))
                # and get the mean
                df[Symbol(alg)][i] = missmean(collect(skipmissing(dfx[:auroc])))
            catch e
                if !isa(e, ArgumentError)
                    warn(e)
                end
            end    
        end
    end
    
    return df
end