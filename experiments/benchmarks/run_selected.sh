for iter in {1..10}
do
	julia -p $1 run_benchmark_selected.jl $iter $2
done