# original data path
datapath = "/opt/output/anomaly"
# where the data will be extracted to
extpath = "/opt/output/extracted"

# this extracts only the important data
include("extract_data.jl")

#### processing of the experiment data #####
# where the data (extracted or original) is stored
data_path = extpath

# where the individual experiment result will be stored
outpath = "/opt/output/output"

# where the summary tables will be stored
evalpath = "/opt/output/eval"

# this produces the evaluation summaries
include("evaluate_experiment.jl")