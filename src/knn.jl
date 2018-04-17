"""
kNN model structure for anomaly detection. Uses brute force distance computation,
that should be changed for better performance. Anomaly score is computed as 
an average of distances to k nearest neighbours.
"""
mutable struct kNN
    k
    data::Array{Float, 2}
    metric # distance metric
    distances # all/last
    contamination # contamination rate
    threshold # decision threshold for classification
    reduced_dim # if this is true, reduce dimensionality of data if > 10 (PCA)
    PCA # tuple containing (P, mu) of the PCA
    fitted
    Beta::Float
end

"""
   kNN(k, contamination, [metric, distances, reduced_dim, threshold, Beta])

kNN model constructor. 

k - number of nearest neighbours taken into consideration
contamination - ratior of contaminated to all samples
metric - for distance computation, using Distances package
distances - use average of all k-nearest or just the kth-nearest neighbour distance
reduced_dim - if this is true, for problems with dim > 10 this is reduced with PCA
threshold - classification threshold
Beta - for automatic threshold computation
"""
function kNN(k::Int, contamination::Real; metric = Euclidean(), distances = "all", 
        threshold = 0.0, reduced_dim = false, Beta = 1.0)
    @assert distances in ["all", "last"]
    return kNN(k, Array{Float,2}(0,0), metric, distances, contamination, threshold,
        reduced_dim, (Array{Float,2}(0,0), Array{Float,1}(0)), false, Beta)
end

"""
    fit!(knn, X)

Fit method for kNN.
"""
function fit!(knn::kNN, X)
    X=Float.(X)
    if knn.reduced_dim
        m = fit(PCA, X, maxoutdim = 10)
        knn.data = transform(m,X)
        knn.PCA = (projection(m), mean(m))
    else
        knn.data = X
    end
    knn.fitted = true
    if size(X,2) < knn.k
        warn("k is higher than data dimension, setting lower...")
        knn.k = size(X,2)
    end
end

"""
    anomalyscore(knn, X, k)

Computes the anomaly score for X using k nearest neighbours.
"""
function anomalyscore(knn::kNN, X::Array{Float,2}, k)
    if !knn.fitted
        error("Call fit!(model, X, Y) before predict can be used!")
    end
    
    # if needed, perform the PCA
    if knn.reduced_dim
        _X = knn.PCA[1]'*(X .- knn.PCA[2])
    else
        _X = X
    end
    
    # compute the distance matrix
    M, N = size(_X)
    
    # now create the output vector of labels
    ascore = Array{Float, 1}(N)
    for n in 1:N
        dn = vec(pairwise(knn.metric, reshape(_X[:,N], M, 1), knn.data))
        if knn.distances == "last"
            ascore[n] = sort(dn)[k]
        else
            ascore[n] = mean(sort(dn)[1:k])
        end
    end
    
    return ascore
end
anomalyscore(knn::kNN, x::Array{Float, 1}, k) = anomalyscore(knn, reshape(x, size(x,1), 1), k)[1]

"""
    anomalyscore(knn, X)

Computes the anomaly score for X.
"""
anomalyscore(knn::kNN, X) = anomalyscore(knn, X, knn.k)

"""
    classify(knn, x, threshold)

Classify an instance x using the discriminator and a threshold.
"""
classify(knn::kNN, X) = Int.(anomalyscore(knn, X) .> knn.threshold)

"""
    getthreshold(knn, x, [Beta])

Compute threshold for kNN classification based on known contamination level.
"""
function getthreshold(knn::kNN, x)
    N = size(x, 2)
    # get reconstruction errors
    ascore = anomalyscore(knn, x)
    # sort it
    ascore = sort(ascore)
    aN = Int(ceil(N*knn.contamination)) # number of contaminated samples
    # get the threshold - could this be done more efficiently?
    (aN > 0)? (return knn.Beta*ascore[end-aN] + (1-knn.Beta)*ascore[end-aN+1]) : (return ascore[end])
end

"""
    setthreshold!(knn, X)

Set model classification threshold based ratior of labels in Y.
"""
function setthreshold!(knn::kNN, X)
    knn.threshold = getthreshold(knn, X)
end

"""
    predict(knn, X)

Predict labels of X.
"""
function predict(knn::kNN, X)
    return classify(knn, X)
end
