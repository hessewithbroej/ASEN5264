import CSV
import HiddenMarkovModels as hmms
import DataFrames as DF
import Random
import Distributions as dists
import XLSX
include("HelperFunctions.jl")
import .HelperFunctions as hf
import Plots as plt


#trust thresholds for new states. In this example, there are two states, one where 0 <= trust < 0.5, and one where 0.5 <= trust.
states = [0.0, 0.2, 0.4, 0.6]

#these should be tuples of sheet name and column number for the feature you want to use
# features = [("ECG_SDNN",5),("EDA_NumSCRs",3),("EDA_PhasicMax",5),("RSP_MV",5),("RSP_RR",5)]
# features = [("ECG_SDNN",5), ("EDA_NumSCRs",3), ("EDA_PhasicMax",5), ("RSP_MV",5), ("RSP_MV",7), ("RSP_RR",5)]
features = [("RSP_MV",5),("RSP_RR",5)]
data_files = ["C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S1_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S2_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S3_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S4_Features.xlsx"]

data = hf.merge_data(data_files,features)
hf.plot_observation_distributions(states,features,data)


@show trans_guess,dists_guess = hf.create_estimates(states,features,data_files)
init_guess = [0.25,0.25,0.25,0.25]

hmm_guess = hmms.HMM(init_guess, trans_guess, dists_guess)

obs_seq,seq_ends = hf.thread_observations(data_files,features)

@show hmm_est, llh_evolution = hmms.baum_welch(hmm_guess,obs_seq;seq_ends)

colors = ["red","orange","yellow","green"]
labels = ["1", "2", "3", "4"]
shapes = [:circle, :square, :diamond, :cross]

best_state_seq, _ = hmms.viterbi(hmm_est,obs_seq;seq_ends)
tmp = [i for i in 1:DF.nrow(data)]

best_seqs = Vector{Vector{Int64}}()
#deconcatenate sequences
for i=1:length(seq_ends)
    start,stop = hmms.seq_limits(seq_ends,i)
    push!(best_seqs,best_state_seq[start:stop])
end

plt_viterbi = plt.scatter([],[])
pop!(plt_viterbi.series_list)
for j=1:length(unique(data.SessionID))
    data_subset = data[data.SessionID.=="S$(j)",:]
    for i=1:length(states)
        inds = findall(x -> x==i, best_seqs[j])
        x = tmp[inds]
        y = data_subset.Trust[inds]
        plt.scatter!(x,y,color=colors[i],markershape=shapes[j])

    end
end
display(plt_viterbi)
