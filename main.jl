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
states = [0.0, 0.5]

#these should be tuples of sheet name and column number for the feature you want to use
# features = [("ECG_SDNN",5), ("RSP_RR",5)]

data_files = ["C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S1_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S2_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S3_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S4_Features.xlsx"]

data = hf.merge_data(data_files,features)

@show trans_guess,dists_guess = hf.create_estimates(states,features,data_files)
init_guess = [0.5, 0.5]

hmm_guess = hmms.HMM(init_guess, trans_guess, dists_guess)

obs_seqs = hf.thread_observations(data_files,features)

@show hmm_est, llh_evolution = hmms.baum_welch(hmm_guess,obs_seq)

colors = ["green", "red"]
labels = ["Low", "High"]

best_state_seq, _ = hmms.viterbi(hmm_est,obs_seq)
plt_viterbi = plt.scatter([],[])
pop!(plt_viterbi.series_list)
for i=1:2
    inds = findall(x -> x==i, best_state_seq)
    y = data.Trust[inds]
    plt.scatter!(y,color=colors[i],label=labels[i])

end
display(plt_viterbi)
