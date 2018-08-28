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
	epochsize
	batchsize
	iter
	buffer
end

function EpochSampler(X::Matrix, nepochs::Int, batchsize::Int)
	M,N = size(X) 
	batchsize = checkbatchsize(N,batchsize,false)
	return EpochSampler(X,M,N,nepochs,Int(ceil(N/batchsize)),batchsize,0,
		sample(1:N,N,replace = false))
end

function next!(s::EpochSampler)
	if s.iter < s.nepochs
		L = length(s.buffer)
		if  L > s.batchsize
			inds = s.buffer[1:s.batchsize]
			s.buffer = s.buffer[s.batchsize+1:end]
		else
			inds = s.buffer
			s.buffer = sample(1:s.N,s.N,replace = false)
			s.iter += 1
		end
		return s.data[:,inds]
	else
		return nothing
	end
end

function enumerate(s::EpochSampler)
	return [(i,next!(s)) for i in 1:(s.nepochs*s.epochsize)]
end

function reset!(s::EpochSampler)
	s.iter = 0
	s.buffer = sample(1:s.N,s.N,replace=false)
end
