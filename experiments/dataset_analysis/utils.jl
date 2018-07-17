using AnomalyDetection, MultivariateStats, TSne, StatsBase
import Base.cat

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
    show()
end