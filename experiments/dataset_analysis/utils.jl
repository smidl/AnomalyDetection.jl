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
    nDtsne(X, n; [args, kwargs])

Returns an n-dimensional representation of X using a PCA transform.
The arguments args and kwargs respond to the TSne.tsne function arguments.
"""
function nDtsne(X, n; args = [0,1000,15], kwargs = [:verbose => true, :progress => true])
    M,N = size(X)
    uN = N # no. of used samples
    while true
        try
            println("sampling $uN samples")
            Y = tsne(X[:,sample(1:N, uN, replace = false)]',n, args...; kwargs...)'
            return Y
        catch e
            if typeof(e)==OutOfMemoryError
                println("$uN samples are too many, getting out of memory error")
                uN = Int(round(uN/2))
            else
                throw(e)
            end
        end
    end
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
        return uncat(nDtsne(X, 2), inds)
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