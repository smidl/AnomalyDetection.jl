mutable struct UniformSampler
	data
	M
	N
	niter
	batchsize
	iter
	replace
end

function checkbatchsize(N,batchsize,replace)
	if batchsize > N && !replace
		warn("batchsize too large, setting to $N")
		batchsize = N
	end
	return batchsize
end

function UniformSampler(X::Matrix, niter::Int, batchsize::Int; replace = false)
	M,N = size(X)
	batchsize = checkbatchsize(N,batchsize,replace)
	return UniformSampler(X,M,N,niter,batchsize,0, replace)
end

function next!(s::UniformSampler)
	if s.iter < s.niter
		s.iter += 1
		return s.data[:,sample(1:s.N,s.batchsize,replace=s.replace)]
	else
		return nothing
	end
end

function reset!(s::UniformSampler)
	s.iter = 0
end

function enumerate(s::UniformSampler)
	return [(i,next!(s)) for i in 1:s.niter]
end

mutable struct EpochSampler
	data
	M
	N
	nepochs
	batchsize
	iter
end

EpochSampler(X::Matrix, nepochs::Int, batchsize::Int) = 
	EpochSampler(X,size(X,1),size(X,2),nepochs,batchsize,0)


