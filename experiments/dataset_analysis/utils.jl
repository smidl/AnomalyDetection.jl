using AnomalyDetection, MultivariateStats, TSne, StatsBase
using PyPlot
import Base.cat

include(joinpath(@__DIR__,"ffs_util.jl"))
"""
    cat(bs::Basicset)

Return an array consisting of all concatenated arrays in bs and 
indices identifying the original array boundaries.
"""
function cat(bs::Basicset)
    X = bs.normal
    inds = [size(X,2)]
    for field in filter(x -> x != :normal, fieldnames(bs))
        x = getfield(bs,field)
        m = size(x,2)
        if m!= 0
            X = cat(2,X,x)
        end
        push!(inds, m)
    end
    return X, inds
end

"""
    uncat(X, inds)

Return a Basicset instance created from X with array boundaries indicated
in inds.
"""
function uncat(X, inds)
    cinds = cumsum(inds)
    return Basicset(
            X[:,1:cinds[1]], 
            X[:,cinds[1]+1:cinds[2]],
            X[:,cinds[2]+1:cinds[3]],
            X[:,cinds[3]+1:cinds[4]],
            X[:,cinds[4]+1:cinds[5]])
end

"""
    nDpca(X, n)

Returns an n-dimensional representation of X using a PCA transform.
"""
nDpca(X, n) = transform(fit(PCA,X,maxoutdim=n),X)

"""
    nDtsne(X, n; [max_samples, args, kwargs])

Returns an n-dimensional representation of X using a PCA transform.
The arguments args and kwargs respond to the TSne.tsne function arguments.
The second return variable are the indices of sampled samples.
"""
function nDtsne(X, n, reduce_dims = 0, max_iter = 1000, perplexity = 15.0;
    max_samples = 1000, verbose = true, progress = true, kwargs...)
    M,N = size(X)
    uN = min(N,max_samples) # no. of used samples
    println("sampling $uN samples")
    sinds = sort(sample(1:N, uN, replace = false))
    Y = tsne(X[:,sinds]',n, reduce_dims, max_iter, perplexity;
                verbose = verbose, progress = progress, kwargs...)'
    return Y, sinds
end

"""
    partition(xinds, sinds)

Compute number of samples in individual groups defined by original group indices
xinds and sample indices sinds.
"""
function partition(xinds, sinds)
    cxinds = [0; cumsum(xinds)]
    a = [length(sinds[cxinds[i] .< sinds .<= cxinds[i+1]]) for 
            i in 1:length(cxinds)-1]
    return a
end

"""
    savetxt(bs::Basicset, path)

Saves a Basicset to the folder "path" into individual .txt files.
"""
function savetxt(bs::Basicset, path)
    mkpath(path)
    for field in fieldnames(bs)
        x = getfield(bs, field)
        if size(x,2) != 0
            writedlm(string(joinpath(path, String(field)), ".txt"),x')
        end
    end
end

"""
    dataset2D(bs::Basicset, variant = ["pca", "tsne"], normalize = true)

Transforms a Basicset into 2D representation using PCA or tSne. 
"""
function dataset2D(bs::Basicset, variant = "pca", normalize = true)
    (variant in ["pca", "tsne"])? nothing : error("variant must be one of [pca, tsne]")
    X, inds = cat(bs)
    (normalize)? X = AnomalyDetection.normalize(X) : nothing
    if variant == "pca"
        return uncat(nDpca(X, 2), inds)
    else
        _X, sinds = nDtsne(X,2;max_samples=1000)
        _inds = partition(inds, sinds)
        return uncat(_X, _inds)
    end
end

"""
    dataset2D(inpath, outpath, variant = ["pca", "tsne"], normalize = true)

Transforms a dataset 
"""
function dataset2D(inpath, outpath, variant = "pca", normalize = true)
    (variant in ["pca", "tsne"])? nothing : error("variant must be one of [pca, tsne]")
    dataset = Basicset(inpath)
    _dataset = dataset2D(dataset, variant, normalize)
    savetxt(_dataset, outpath)
    return _dataset
end

"""
    X = original array
    Y = output array by tsne
"""
function getLSmatrices(X, Y)
    MY, N = size(Y)
    _y = reshape(Y, MY*N);
    M,N = size(X)
    _X = zeros(N*2, 2*M+1)
    _X[:,end] = ones(N*2)
    for n in 1:N
        x = X[:,n]'
        _X[2*n-1,1:M] = x
        _X[2*n,M+1:end-1] = x
    end
    return _y, _X
end

"""
    tsneLS(X,Y,bias=true)

Given an original data matrix X and and a matrix Y transformed
from X using tSne, produce the linear mapping Y = A*X + b.
"""
function tsneLS(X,Y)
    M,N = size(X)
    _y, _X = getLSmatrices(X,Y)
    a = llsq(_X, _y,bias=false)
    b = a[end]
    A = [a[1:M]';a[M+1:end-1]']
    return A, b
end

pred(X,A,b) = A*X + b

"""
   scatter_data(data,t="")

Scatter a Basicset groups. 
"""
function scatter_data(data,t="";kwargs...)
    figure()
    for field in fieldnames(data)
        X = getfield(data, field)
        if length(X) > 0
            scatter(X[1,:],X[2,:],label=string(field);kwargs...)
        end
    end
    legend()
    title(t)
end

################################################
## this is for plotting and other evaluation ###
################################################
"""
    Xy(data)

Concatenate the training and testing dataset.
"""
Xy(data) = (cat(2,data[1].data,data[2].data),
            cat(1,data[1].labels,data[2].labels))

"""
    scatterbinary(X,y;kwargs...)

Scatter normal and anomalous samples.
"""
function scatterbinary(X,y;kwargs...)
    if length(y[y.==0]) > 0
        scatter(X[1,y.==0],X[2,y.==0],label="normal data",c="g";kwargs...)
    end
    if length(y[y.==1]) > 0
        scatter(X[1,y.==1],X[2,y.==1],label="anomalous data",c="r";kwargs...)
    end
end

"""
   scatteralldata(data,tit)

Scatter all the points in (trdata, tstdata) tuple. 
"""
function scatteralldata(data;kwargs...)
    X,y = Xy(data)
    @assert size(X,1) == 2
    scatterbinary(X,y;kwargs...)
end

"""
    plot_ffs_overview(data,tit,loc="")

Scatter all the points, give title and save to loc. 
"""
function plot_ffs_overview(data,tit,loc="")
    scatteralldata(data,s=10)
    title(tit)
    legend()
    if loc !=""
        f = "$(dataset)_$(pair[1])_$(pair[2])_all_scatter.png"
        savefig(joinpath(loc,f))
    end
end

"""
    ascontours(model,xl,yl,griddensity=30)

Return anomaly score of model on a grid specified by xl, yl and density.
"""
function ascontours(model,xl,yl,griddensity=30)
    # compute the anomaly score on a grid
    x = linspace(xl[1], xl[2], griddensity)
    y = linspace(yl[1], yl[2], griddensity)
    zz = zeros(size(y,1),size(x,1))
    for i in 1:size(y, 1)
        for j in 1:size(x, 1)
            zz[i,j] = (:encoder in fieldnames(model))?
             AnomalyDetection.anomalyscore(model, AnomalyDetection.Float.([x[j], y[i]]), 10):
            AnomalyDetection.anomalyscore(model, AnomalyDetection.Float.([x[j], y[i]]))
        end
    end
    return x,y,zz
end

"""
    xylims(X)

Compute limits for 2D plot from array X.
"""
function xylims{T<:Real}(X::AbstractArray{T,2})
    xl = [minimum(X[1,:]), maximum(X[1,:])]
    yl = [minimum(X[2,:]), maximum(X[2,:])]
    dx = 0.05*abs(xl[2]-xl[1])
    dy = 0.05*abs(yl[2]-yl[1])
    xl[1] -= dx
    xl[2] += dx
    yl[1] -= dy
    yl[2] += dy
    return xl, yl
end

"""
    xylims(data)

Compute limits for 2D plot from (trdata,tstdata) tuple.
"""
function xylims(data)
    X,y = Xy(data)
    @assert size(X,1) == 2
    return xylims(X)
end


"""
    scattertraindata(data)

Scatter training samples.
"""
function scattertraindata(data;kwargs...)
    X = data[1].data
    y = data[1].labels
    scatter(X[1,y.==0],X[2,y.==0],label="training data",c="g";kwargs...)
end

"""
    plot_contour_train(model,data,tit,loc="")

Plot AS score contours and training data.
"""
function plot_contour_train(model,data,tit,loc="")
    xl, yl = xylims(data)
    x,y,zz = ascontours(model,xl,yl,50)
    c = contourf(x,y,zz,100)
    colorbar()
    scattertraindata(data,s=10)
    title(tit)
    legend()
    if loc !=""
        if (:encoder in fieldnames(model))
            s = "vae"
        else
            s = "knn"
        end
        f = "$(dataset)_$(pair[1])_$(pair[2])_as_contour_atrain_$(s).png"
        savefig(joinpath(loc,f))
    end
end

"""
    labels(model,data)

Get labels with automatically computed threshold.
"""
function labels(model,data)
    X,ytrue = data[2].data, data[2].labels
    M,N = size(X)
    as = (:encoder in fieldnames(model))?
        AnomalyDetection.anomalyscore(model, X, 50):
        AnomalyDetection.anomalyscore(model, X)
    auc = EvalCurves.auc(EvalCurves.roccurve(as,ytrue)...)
    cont = length(ytrue[ytrue.==1])/length(ytrue)
    sas = sort(as)
    aN = Int(ceil(N*cont)) # number of contaminated samples
    tr = ((aN > 0)? (sas[end-aN] + sas[end-aN+1])/2 : (sas[end]))
    yhat = Int.(as .> tr)
    return yhat, auc
end

"""
    scatterbinary(X,ytrue,yhat;kwargs...)

Scatter tru,false positives and true,false negatives.
"""
function scatterbinary(X,ytrue,yhat;kwargs...)
    itn = tn(ytrue,yhat)
    scatter(X[1,itn],X[2,itn],
        c="g",label="true negative";kwargs...)
    ifp = fp(ytrue,yhat)
    scatter(X[1,ifp],X[2,ifp],
        c="r",label="false positive";kwargs...)
    itp = tp(ytrue,yhat)
    scatter(X[1,itp],X[2,itp],
        c="w",label="true positive";kwargs...)
    ifn = fn(ytrue,yhat)
    scatter(X[1,ifn],X[2,ifn],
        c="k",label="false negative";kwargs...)
end

tp(ytrue,yhat) = (ytrue.==1) .& (yhat.==1)
fp(ytrue,yhat) = (ytrue.==0) .& (yhat.==1)
tn(ytrue,yhat) = (ytrue.==0) .& (yhat.==0)
fn(ytrue,yhat) = (ytrue.==1) .& (yhat.==0)

ntp(ytrue,yhat) = sum(tp(ytrue,yhat))
nfp(ytrue,yhat) = sum(fp(ytrue,yhat))
ntn(ytrue,yhat) = sum(tn(ytrue,yhat))
nfn(ytrue,yhat) = sum(fn(ytrue,yhat))

"""
    plot_contour_fit(model,data,tit,loc="")

Plot AS contours and fit to testing data.
"""
function plot_contour_fit(model,data,tit,loc="")
    X,ytrue = data[2].data, data[2].labels
    yhat, auc = labels(model,data)
    tpr = ntp(ytrue,yhat)/sum(yhat)
    fpr = nfp(ytrue,yhat)/sum(abs.(1-yhat))
    xl,yl = xylims(X)
    x,y,zz = ascontours(model,xl,yl,50)
    contourf(x,y,zz,100)
    colorbar()
    scatterbinary(X,ytrue,yhat,s=10)
    tit = string(tit, "testing data, AUC = $(round(auc,3))")
    tit = string(tit, "\nTPR = $(round(tpr,3)), FPR = $(round(fpr,3))")
    title(tit)
    legend()
    if loc !=""        
        if (:encoder in fieldnames(model))
            s = "vae"
        else
            s = "knn"
        end
        f = "$(dataset)_$(pair[1])_$(pair[2])_as_contour_fit_$(s).png"
        savefig(joinpath(loc,f))
    end
end

"""
    lineinfo(df,iline)
    
Get info from the line iline of df.
"""
function lineinfo(df,iline)
    dataset = String(df[:dataset][iline])
    pair = [df[:f1][iline],df[:f2][iline]]
    vs = round(df[:vae][iline],3)
    ks = round(df[:knn][iline],3)
    k = Int(df[:k][iline])
    return dataset, pair, vs, ks, k
end

"""
    plot_all(data,k,tit="")

Plot a 3x2 plot of results.
"""
function plot_all(data,k,tit="")
    trdata=data[1]
    tstdata=data[2]
    
    # plot overview of the features
    figure(figsize=(15,20))

    # hide the first plot
    subplot(321)
    plt[:axis]("off")

    # shift and plot the second one
    f=subplot(322)
    a = f[:get_position]()
    a[:x0] -= 0.211
    a[:x1] -= 0.211
    f[:set_position](a)
    plot_ffs_overview(data,tit)

    # train models
    ka, knn, _ = knnscore(trdata,tstdata,k)
    va, vae, _ = vaescore(trdata,tstdata)
    
    # plot as and training data
    t = "kNN (k=$(k)) anomaly score contours\ntraining data"
    subplot(323)
    plot_contour_train(knn,data,t)
    t = "VAE anomaly score contours\ntraining data"
    subplot(324)
    plot_contour_train(vae,data,t)    
    ax = gca()
    ax[:legend_][:remove]()

    # plot
    t = ""
    subplot(325)
    plot_contour_fit(knn,data,t)
    t = ""
    subplot(326)
    plot_contour_fit(vae,data,t)
    ax = gca()
    ax[:legend_][:remove]()
end

"""
   plot_ffs_all(df,iline,variant,loc="",showfig=false)

Plot a 3x2 grid containing all the important information using df from the findfeatures experiment. 
"""
function plot_ffs_all(df,iline,variant,loc="",showfig=false;seed=NaN)
    # get the information from a line in df
    dataset, pair, vs, ks, k = lineinfo(df,iline)
    
    # setup other stuff
    alldata = ((variant == "some")? false : true) 
    if loc != ""
        mkpath(loc)
    end
    
    # get the data
    data = getdata(dataset,alldata,(isnan(seed))?518:seed);
    data=(subfeatures(data[1], pair),subfeatures(data[2], pair)) 
    
    # do the plots
    t = "findfeature.jl, $dataset$pair, all data,\n vae:$vs, knn:$ks"
    plot_all(data,k,t)

    # savefig
    if loc != ""
        if isnan(seed)
            f = joinpath(loc,"$(dataset)_$(pair[1])_$(pair[2]).png")
        else
            f = joinpath(loc,"$(dataset)_$(pair[1])_$(pair[2])_$(seed).png")
        end
        savefig(f,bbox_inches="tight")
    end

    if showfig
        show()
    end
end

"""
   plot_all(dataset,inpath,loc="",showfig=false)

Plot a 3x2 grid containing all the important information from given experimental data. 
\ninpath - where data is stored
\nloc - where data are to be saved
"""
function plot_general_all(dataset,inpath,tit,loc="",showfig=false;seed=NaN)
    # setup other stuff
    alldata = false
    k = 1
    if loc != ""
        mkpath(loc)
    end
    
    # get the data
    data = getdata(dataset,alldata,(isnan(seed))?518:seed,loc=inpath);
    
    # do the plots
    t = "$tit, $dataset, all data"
    plot_all(data,k,t)

    # savefig
    if loc != ""
        if isnan(seed)
            f = joinpath(loc,"$(dataset).png")
        else
            f = joinpath(loc,"$(dataset)_$(seed).png")
        end
        savefig(f,bbox_inches="tight")
    end

    if showfig
        show()
    end
end

"""
    plot_ffs_overview(dataset,pair,alldata,tit,loc="")

Scatter all the points given dataset name and pair of indices.
"""
function plot_ffs_overview(dataset::String,pair,alldata,tit,loc="")
    @assert length(pair) == 2
    data = subfeatures(getdata(dataset,alldata),pair)
    plot_ffs_overview(data,tit,loc)
end

"""
    plot_contour_train(model,dataset,pair,alldata,tit,loc="")

Plot AS contours and training data.
"""
function plot_contour_train(model,dataset::String,pair,alldata,tit,loc="")
    data = getdata(dataset,alldata);
    data=(subfeatures(data[1], pair),subfeatures(data[2], pair)) 
    
   return plot_contour_train(model,data,tit,loc) 
end

"""
    plot_contour_fit(model,dataset,pair,alldata,tit,loc="")

Plot AS contours and fit to testing data.
"""
function plot_contour_fit(model,dataset::String,pair,alldata,tit,loc="")
    data = getdata(dataset,alldata);
    data=(subfeatures(data[1], pair),subfeatures(data[2], pair)) 
    return plot_contour_fit(model,data,tit,loc)
end
