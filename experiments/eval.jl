using DataFrames, Query, FileIO, ValueHistories
import Missings: missing, skipmissing, ismissing

push!(LOAD_PATH, "../src")
using AnomalyDetection

"""
    auroc(ascore, labels, [weights])

Compute area under ROC curve. If ascores are NaNs, returns value.
"""
function auroc(ascore, labels, weights = "same")
    if isnan(ascore[1])
        return missing
    else
        tprvec, fprvec = AnomalyDetection.getroccurve(ascore, labels)
        return AnomalyDetection.auc(fprvec, tprvec, weights)
    end
end

"""
    topprecision(ascore, labels, p)

Computes precision in prediction based on top p% rated instances.
"""
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

"""
    computedatasetstats(datapath, dataset, algnames)

Compute comprehensive stats for a single dataset and all experiments that were run on it.
Returns a DataFrame containing training and testing auroc, top 5% precision
and fit/predict time.
"""
function computedatasetstats(datapath, dataset, algnames)
    df = DataFrame()
    cnames = ["dataset", "algorithm", "iteration", "settings", "train_auroc", "test_auroc", 
        "train_aauroc", "test_aauroc", "top_5p", "top_1p", "fit_time", "predict_time"]
    for name in cnames
        df[Symbol(name)] = Any[]
    end
    
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
                tra = auroc(load(f, "training_anomaly_score"), load(f, "training_labels"))
                tsta = auroc(load(f, "testing_anomaly_score"), load(f, "testing_labels"))
               
                # testing and training augmented auroc
                traa = auroc(load(f, "training_anomaly_score"), load(f, "training_labels"), "1/x")
                tstaa = auroc(load(f, "testing_anomaly_score"), load(f, "testing_labels"), "1/x")
                 
                # compute top 5% ascore samples precision
                tp5 = topprecision(load(f, "training_anomaly_score"), load(f, "training_labels"), 0.05)
                tp1 = topprecision(load(f, "training_anomaly_score"), load(f, "training_labels"), 0.01)

                # extract the times as well
                ft = load(f, "fit_time")
                pt = load(f, "predict_time")

                # save the data
                push!(df, [dataset, alg, iter, io, tra, tsta, traa, tstaa, tp5, tp1, ft, pt])
            end
        end
    end

    return df
end

"""
    computestats(datapath, algnames)

Gather comprehensive statistics for all datasets in a single dataframe.
"""
function computestats(datapath, algnames)
    df = DataFrame()
    cnames = ["dataset", "algorithm", "iteration", "settings", "train_auroc", "test_auroc",
         "train_aauroc", "test_aauroc", "top_5p", "top_1p", "fit_time", "predict_time"]
    for name in cnames
        df[Symbol(name)] = Any[]
    end
    
    datasets = readdir(datapath)

    for dataset in datasets
        df = [df; computedatasetstats(datapath, dataset, algnames)]
    end
    
    return df
end

"""
    loadtable(fname, datacols)

Load a csv file into DataFrame, reformatting specified data columnsto floats and missings.
"""
function loadtable(fname, datacols)
    #load the df
    data = readtable(fname)
    
    for cname in names(data)
        data[cname] = Array{Any,1}(data[cname])
    end
    
    # round the nubmers and replace "missing" with actual missing values
    rounddf!(convertdf!(data, datacols), 6, datacols)
    
    return data
end

"""
    convertdf!(df, datacols)

Convert cells of df to floats or missings - specially crafted for dfs loaded from csvs.
"""
function convertdf!(df, datacols)
    nrows, ncols = size(df)
    
    # go through the whole df and replace missing strings with actual Missing type
    # and floats with float
    if typeof(datacols) == Int64
        cnames = names(df)[datacols:end]
    else
        cnames = names(df)[datacols]
    end

    for cname in cnames
        for i in 1:nrows
            (df[cname][i] == "missing")? df[cname][i]=missing : 
                df[cname][i]=float(df[cname][i])
        end
    end

    return df
end

"""
    convertdf(df, datacols)

Convert cells of df to floats or missings - specially crafted for dfs loaded from csvs.
"""
function convertdf(df, datacols)
    _df = deepcopy(df)
    return convertdf!(_df)
end

"""
   rounddf!(df, n, datacols)

Round values in datacols of df to n valid digits.
"""
function rounddf!(df, n, datacols)
    nrows, ncols = size(df)
    
    # go through the whole df and replace missing strings with actual Missing type
    # and floats with float
    if typeof(datacols) == Int64
        cnames = names(df)[datacols:end]
    else
        cnames = names(df)[datacols]
    end

    for cname in cnames
        for i in 1:nrows
            (ismissing(df[cname][i]))? df[cname][i]=missing : 
                df[cname][i]=round(float(df[cname][i]),n)
        end
    end

    return df
end

"""
   rounddf(df, n, datacols)

Round values in datacols of df to n valid digits.
"""
function rounddf(df, n, datacols)
    _df = deepcopy(df)
    return rounddf!(_df, n, datacols)
end

"""
    rankdf(df, [rev])

Compute row ranks for a DataFrame and add bottom line with mean ranks.
Ties receive average rank.
rev (default true) - higher score is better 
"""
function rankdf(df, rev = true)
    _df = deepcopy(df)
    nrows, ncols = size(_df)
    nalgs = ncols - 1
    
    algnames = names(df)[2:end]
    
    for i in 1:nrows
        row = _df[i,2:end]
        arow = reshape(Array(row), nalgs)
        isort = sortperm(arow, rev = rev)
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
        _df[alg][end] = missmean(_df[alg][1:end-1])
    end
    
    return _df
end

"""
    missmean(x)

If x is empty, return missing, else compute mean.
"""
function missmean(x)
    if size(x,1) == 0
        return missing
    else
        return mean(collect(skipmissing(x)))
    end
end

"""
    missmax(x)

If x is empty, return missing, else return maximum.
"""
function missmax(x)
    if size(x,1) == 0
        return missing
    else
        return maximum(collect(skipmissing(x)))
    end
end

"""
    missfindmax(x)

If x is empty, return missing, else return maximum and its indice.
"""
function missfindmax(x)
   if size(x,1) == 0
        return missing
    else
        return findmax(collect(skipmissing(x)))
    end
end 


"""
    collectscores(outpath, algs, scoref)

Collect scores on datasets in outpath, for specified algorithm and score function.
Raturns a DataFrame.
"""
function collectscores(outpath, algs, scoref)
    df = createdf(algs)
    nalgs = size(algs,1)
    
    # collect dataset names
    datasets = readdir(outpath)
    
    for dataset in datasets
        path = joinpath(outpath, dataset) 
        
        # create a row per dataset
        row = Array{Any,1}(nalgs+1)
        row[2:end] = missing
        row[1] = dataset
        for (n,alg) in zip(1:nalgs,algs)
            try
                f = joinpath(path, "$(alg).csv")
                dfx = scoref(loadtable(f, 5), [alg])
                row[n+1] = dfx[Symbol(alg)][1] 
            catch e
                nothing
            end
        end

        push!(df, reshape(row, 1, nalgs+1))
        
    end
    
    return df
end

"""
    createdf(algs)

Create an empty DataFrame with one dataset column and x algorithm columns.
"""
function createdf(algs)
    df = DataFrame()
    df[:dataset] = String[]
    for alg in algs
        df[Symbol(alg)] = Any[]
    end

    return df
end

"""
    preparedf(data, algs)

Prepare a DataFrame for comparing algorithms across datasets.
"""
function preparedf(data, algs)
    df = createdf(algs)

    nalgs = size(algs,1)
    dataset = data[:dataset][1]
    
    row = Array{Any,1}(nalgs+1)
    row[2:end] = missing
    row[1] = dataset
    push!(df, reshape(row, 1, nalgs+1))
    
    return df, dataset
end

"""
    maxauroc(data, algs, [auc_type])

Score algorithms according to their maximum auroc on a testing dataset 
averaged over experiment iterations.
"""
function maxauroc(data, algs, auc_type = "normal")
    df, dataset = preparedf(data, algs)

    # auc_type for augmented or normal auc
    if auc_type == "normal"
        tstsym = :test_auroc
    elseif auc_type == "augmented"
        tstsym = :test_aauroc
    end

    for alg in algs
        dfx = @from r in data begin
            @where r.algorithm == alg && r.dataset == dataset
            @select {r.iteration, getfield(r, tstsym)}
            @collect DataFrame
        end
        rename!(dfx, :_2_, tstsym)
            
        try
            # group this by iterations
            dfx = by(dfx, [:iteration], 
                d -> DataFrame(auroc = 
                    missmax(d[tstsym])))
            # and get the mean
            df[Symbol(alg)][1] = round(missmean(dfx[:auroc]),6)
        catch e
            if !isa(e, ArgumentError)
                nothing #warn(e)
            else
                throw(e)
            end
        end    
    end

    return df
end

"""
    trainauroc(data, algs, [auc_type])

Choose algorithm with parameters according to maximum auroc on training dataset,
then compute the score as average of testing auroc with these parameters over iterations.
"""
function trainauroc(data, algs, auc_type = "normal")
    df, dataset = preparedf(data, algs)

    # auc_type for augmented or normal auc
    if auc_type == "normal"
        trsym = :train_auroc
        tstsym = :test_auroc
    elseif auc_type == "augmented"
        trsym = :train_aauroc
        tstsym = :test_aauroc
    end

    for alg in algs
        dfx = @from r in data begin
            @where r.algorithm == alg && r.dataset == dataset
            @select {r.settings, r.iteration, getfield(r, trsym), getfield(r,tstsym)}
            @collect DataFrame
        end
        rename!(dfx, :_3_, trsym)
        rename!(dfx, :_4_, tstsym)
        
        try
            # mean aggregate it by settings
            traindf = by(dfx, [:settings],
                            d -> DataFrame(train_auroc = 
                            missmean(d[trsym])))
            # get the best settings
            sort!(traindf, cols = :train_auroc, rev = true)
            topalg = ""
            for j in 1:size(traindf,1)
                if !ismissing(traindf[:train_auroc][j])
                    topalg = traindf[:settings][j]
                    break
                end
            end
            # and get the mean of the best setting test auroc over all iterations
            testdf = @from r in dfx begin
                     @where r.settings == topalg
                     @select {r.settings, r.iteration, getfield(r, tstsym)}
                     @collect DataFrame
            end
            rename!(testdf, :_3_, tstsym)
            df[Symbol(alg)][1] = round(missmean(testdf[tstsym]),6)
        catch e
            if !isa(e, ArgumentError)
                throw(e)
                nothing
            else
                throw(e)
            end
        end    
    end
    
    return df

end

"""
    toprec(data, algs, [label, auc_type])

Choose algorithm with parameters according to precision on top x% instances in training dataset,
then compute the score as average of testing auroc with these parameters over iterations.
"""
function topprec(data, algs, label = :top_5p, auc_type = "normal")
    df, dataset = preparedf(data, algs)

    # auc_type for augmented or normal auc
    if auc_type == "normal"
        tstsym = :test_auroc
    elseif auc_type == "augmented"
        tstsym = :test_aauroc
    end

    for alg in algs
        dfx = @from r in data begin
            @where r.algorithm == alg && r.dataset == dataset
            @select {r.settings, r.iteration, getfield(r, label), getfield(r, tstsym)}
            @collect DataFrame
        end
        rename!(dfx, :_3_, label)
        rename!(dfx, :_4_, tstsym)
    
        try
            # mean aggregate it by settings
            traindf = by(dfx, [:settings],
                            d -> DataFrame(top = 
                            missmean(d[label])))
            # get the best settings
            sort!(traindf, cols = :top, rev = true)
            topalg = ""
            for j in 1:size(traindf,1)
                if !ismissing(traindf[:top][j])
                    topalg = traindf[:settings][j]
                    break
                end
            end
            # and get the mean of the best setting test auroc over all iterations
            testdf = @from r in dfx begin
                     @where r.settings == topalg
                     @select {r.settings, r.iteration, getfield(r, tstsym)}
                     @collect DataFrame
            end
            rename!(testdf, :_3_, tstsym)
            df[Symbol(alg)][1] = round(missmean(testdf[tstsym]),6)
        catch e
            if !isa(e, ArgumentError)
                nothing
            else
                throw(e)
            end
        end    
    end
    
    return df

end

"""
    meantime(data, algs, t)

Score algorithm on a dataset based on mean fit/predict times over iterations and parameter settings.
"""
function meantime(data, algs, t)
    @assert t in ["predict_time", "fit_time"]

    df, dataset = preparedf(data, algs)

    for alg in algs
        dfx = @from r in data begin
            @where r.algorithm == alg && r.dataset == dataset
            @select {r.settings, r.iteration, getfield(r, Symbol(t))}
            @collect DataFrame
        end
        rename!(dfx, :_3_, Symbol(t))

        try
            # mean aggregate it by settings
            a = round(missmean(dfx[Symbol(t)]),6)
            df[Symbol(alg)][1] = a
        catch e
            if !isa(e, ArgumentError)
                throw(e)
            else
                throw(e)
            end
        end    
    end
    
    return df
end

"""
    eol(s)

Replaces the "& " at the end of s with a tabular end of line.
"""
function eol(s)
    return string(s[1:end-2], " \\\\ \n")
end

"""
    wspad(s, n)

Pads s with n white spaces.
"""
function wspad(s, n)
    return string(s, repeat(" ", n))
end

"""
    df2tex(df, caption="", pos = "h", align = "c"; 
           fitcolumn = false, lasthline = false)

Convert DataFrame to a LaTex table.
"""
function df2tex(df, caption="", pos = "h", align = "c"; 
    fitcolumn = false, lasthline = false, firstvline = false)
    cnames = names(df)
    nrows, ncols = size(df)

    # create the table beginning
    if fitcolumn
        s = "\\begin{table} \n \\resizebox{\\columnwidth}{!}{ \n \\begin{tabular}[$pos]{"
    else    
        s = "\\begin{table} \n \\begin{tabular}[$pos]{"
    end
    for n in 1:ncols
        if firstvline && n == 1
            s = string(s, "$align | ")
        else
            s = string(s, "$align ")
        end
    end
    s = string(s,"} \n")

    # create the header
    s = wspad(s,2)
    for name in names(df)
        s = string(s, "$name & ")
    end
    s = eol(s)
    s = wspad(s,2)
    s = string(s, "\\hline \n")
    
    # fill the table
    for i in 1:nrows
        s = wspad(s,2)
        for j in 1:ncols
            s = string(s, "$(df[i,j]) & ")
        end
        s= eol(s)
        if lasthline && i == nrows-1
            s = wspad(s,2)
            s = string(s, "\\hline\n")
        end
    end

    # create the table ending
    s = string(s, " \\end{tabular}\n")
    if fitcolumn
        s = string(s, " }\n")
    end
    if caption!=""
        s = string(s, " \\caption{$caption} \n")
    end
    s = string(s, "\\end{table}")

    return s
end

"""
    string2file(f, s)

Save string s to file f.
"""
function string2file(f, s)
    open(f, "w") do _f
        write(_f, s)
    end
end

"""
    miss2hyphen!(df)

Replaces all missing values with a hyphen "--".
"""
function miss2hyphen!(df)
    nrows, ncols = size(df)

    for i in 1:nrows
        for j in 1:ncols
            if ismissing(df[i,j])
             df[i,j]="--"
            end 
        end
    end

    return df
end

"""
    miss2hyphen(df)

Replaces all missing values with a hyphen "--".
"""
function miss2hyphen(df)
    _df = deepcopy(df)
    return miss2hyphen!(df)
end

"""
    rpaddf!(df,n)

Rightpad all numerical values with zeros to have n decimal digits.
"""
function rpaddf!(df,n)
    nrows, ncols = size(df)

    for i in 1:nrows
        for j in 1:ncols
            if typeof(df[i,j]) <: AbstractFloat
                s = split("$(df[i,j])", ".") 
                df[i,j] = "$(s[1]).$(rpad(s[2],n,"0"))"             
            end 
        end
    end

    return df
end

"""
    rpaddf(df,n)

Rightpad all numerical values with zeros to have n decimal digits.
"""
function rpaddf(df,n)
    _df = deepcopy(df)
    return rpaddf!(_df,n)
end

"""
    mergedfs(ldf, rdf)

Merges DataFrames for the article purpose.
"""
function mergedfs(ldf, rdf)
    nrows, ncols = size(ldf)
    @assert (nrows, ncols) == size(rdf)

    df = deepcopy(ldf)

    # first merge all cells
    for i in 1:nrows
        for j in 2:ncols
            df[i,j] = "$(df[i,j])($(rdf[i,j]))"
        end
    end

    return df
end
