"""
kNN model structure for binary classification. Uses brute force distance computation,
that should be changed for better performance.
"""
mutable struct kNN
    k
    data
    labels 
    metric # distance metric
    weights # uniform, distance
    reduced_dim # if this is true, reduce dimensionality of data if > 10 (PCA)
    PCA # tuple containing (P, mu) of the PCA
    fitted
end

"""
   kNN(k, [metric, weights, reduced_dim])

kNN model constructor. 

k - number of nearest neighbours taken into consideration
metric - for distance computation, using Distances package
weights - uniform/distance
reduced_dim - if this is true, for problems with dim > 10 this is reduced with PCA
"""
function kNN(k::Int; metric = Euclidean(), weights = "uniform", reduced_dim = false)
    @assert weights in ["uniform", "distance"]
    return kNN(k, Array{Float64,2}(0,0), Array{Int64, 1}(0), metric, weights, 
        reduced_dim, (Array{Float64,2}(0,0), Array{Float64,1}(0)), false)
end

"""
    fit!(knn, X, Y)

Fit method for kNN.
"""
function fit!(knn::kNN, X, Y)
    if knn.reduced_dim
        m = fit(PCA, X, maxoutdim = 10)
        knn.data = transform(m,X)
        knn.PCA = (projection(m), mean(m))
    else
        knn.data = X
    end
    knn.labels = Y
    knn.fitted = true
end

"""
    predict(knn, X)

Predict labels of X.
"""
function predict(knn::kNN, X::Array{Float64, 2})
    if !knn.fitted
        println("Call fit!(model, X, Y) before predict can be used!")
        return
    end
    
    # if needed, perform the PCA
    if knn.reduced_dim
        _X = knn.PCA[1]'*(X .- knn.PCA[2])
    else
        _X = X
    end
    
    # compute the distance matrix
    M, N = size(_X)
    dm = pairwise(knn.metric, _X, knn.data)
    
    # now create the output vector of labels
    labels = Array{Int64, 1}(N)
    for n in 1:N
        dn = dm[n,:] 
        if knn.weights == "distance"
            isort = sortperm(dn)
            weights = 1./(dn[isort][1:knn.k] + 1e-3)
            weights = weights/sum(weights)
            labels[n] = Int(round(sum(weights.*(knn.labels[isort][1:knn.k]))))
        else
            labels[n] = Int(round(mean(knn.labels[sortperm(dn)][1:knn.k])))
        end
    end
    
    return labels
end

# 1D method 
predict(knn::kNN, x::Array{Float64,1}) = predict(knn, reshape(x, size(x,1), 1))