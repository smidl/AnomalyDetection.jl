using AnomalyDetection, MultivariateStats
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
	pcaXnD(X, n)

Returns an n-dimensional representation of X using a PCA transform.
"""
function pcaXnD(X, n)
    return transform(fit(PCA,X,maxoutdim=n),X)
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
	data2Dpca(bs::Basicset)

Transforms a Basicset into 2D representation using PCA. 
"""
function data2Dpca(bs::Basicset)
    X, inds = cat(bs)
    return uncat(pcaXnD(X, 2), inds)
end

"""
	data2Dpca(inpath, outpath)

Transforms 
"""
function data2Dpca(inpath, outpath)
    dataset = Basicset(inpath)
    _dataset = data2Dpca(dataset)
    savetxt(_dataset, outpath)
    return _dataset
end