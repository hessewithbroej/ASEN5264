import CSV
import HiddenMarkovModels as hmms
import DataFrames as DF
import Random
import Distributions as dists
import XLSX
include("HelperFunctions.jl")
import .HelperFunctions as hf
import Plots as plt

gt_trust = DF.DataFrame(XLSX.readdata("FeaturesForModelsAFP30.xlsx", "Trust", "A2:A193"), :auto)
gt_trust = Float64[x for x in gt_trust.x1]
truth_state = [num > 0.5 ? 2 : 1 for num in gt_trust]
#trust thresholds for new states. In this example, there are two states, one where 0 <= trust < 0.5, and one where 0.5 <= trust.
states = [0.0, 0.5]

#these should be tuples of sheet name and column number for the feature you want to use
# features = [("ECG_SDNN",5),("EDA_NumSCRs",3),("EDA_PhasicMax",5),("RSP_MV",5),("RSP_RR",5)]
# features = [("ECG_SDNN",5), ("EDA_NumSCRs",3), ("EDA_PhasicMax",5), ("RSP_MV",5), ("RSP_MV",7), ("RSP_RR",5)]
# features = [("Features",3),("Features",22)]   #71fNIRS_Vers4 265
# features = [("fNIRS_Vers4",71),("fNIRS_Vers4",265)]
features_neuro = [("fNIRS_Vers4",18),("fNIRS_Vers4",20),("fNIRS_Vers4",44),("fNIRS_Vers4",45),("fNIRS_Vers4",49),("fNIRS_Vers4",52),("fNIRS_Vers4",77),("fNIRS_Vers4",96),("fNIRS_Vers4",102),("fNIRS_Vers4",107),("fNIRS_Vers4",121),("fNIRS_Vers4",123),("fNIRS_Vers4",124),("fNIRS_Vers4",128),("fNIRS_Vers4",160),("fNIRS_Vers4",225),("fNIRS_Vers4",240),("fNIRS_Vers4",277),("fNIRS_Vers4",290),("fNIRS_Vers4",322),("fNIRS_Vers4",338),("fNIRS_Vers4",345),("fNIRS_Vers4",348),("fNIRS_Vers4",357)]
features_physio = [("Features",2),("Features",11),("Features",20),("Features",29), ("Features",38),("Features",47),("Features",56),("Features",65),("Features",74),("Features",83),("Features",92),("Features",101),("Features",110),("Features",119),("Features",128),("Features",137),("Features",146)]

features = append!(features_neuro,features_physio)
# features = features_physio
data_files = ["FeaturesForModelsAFP30.xlsx"]

# data_files = ["C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S1_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S2_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S3_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S4_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S1_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S2_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S3_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S4_Features.xlsx"]


data = hf.merge_data(data_files,features)
hf.plot_observation_distributions(states,features,data)

@show trans_guess,dists_guess = hf.create_estimates(states,features,data_files)
init_guess = [0.5,0.5]

hmm_guess = hmms.HMM(init_guess, trans_guess, dists_guess)

@show obs_seq,seq_ends = hf.thread_observations(data_files,features)


@show hmm_est, llh_evolution = hmms.baum_welch(hmm_guess,obs_seq;seq_ends)

colors = ["red","green"]
labels = ["1", "2", "3", "4"]
shapes = [:circle, :square, :diamond, :hexagon, :cross, :xcross, :star4, :star6]

best_state_seq, _ = hmms.viterbi(hmm_est,obs_seq;seq_ends)
tmp = [i for i in 1:DF.nrow(data)]

best_seqs = Vector{Vector{Int64}}()
#deconcatenate sequences
for i=1:length(seq_ends)
    start,stop = hmms.seq_limits(seq_ends,i)
    push!(best_seqs,best_state_seq[start:stop])
end

studies = unique(data.SessionID)
plt_obj = plt.scatter( 1:length(truth_state), gt_trust, label=false,color = [value1 == value2 ? "green" : "red" for (value1, value2) in zip(truth_state, best_state_seq) ], xlabel = "Epoch", ylabel = "Ground Truth", title = "Viterbi Classification of Binary Trust Responses", grid = true,ylim=(0,1))
plt.scatter!([],[],color="green",label="Correctly Predicted Trust");
plt.scatter!([],[],color="red",label="Incorrectly Predicted Trust",legendfontsize=10);
display(plt_obj)

