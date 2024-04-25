
import HiddenMarkovModels as hmms
import CSV
import DataFrames as DF
import Random
import Distributions as dists
import Plots as plt
import Dates

Random.seed!(12321)

#load data, add a differential column
data = DF.DataFrame(CSV.File("C:/Users/hesse/Desktop/Code/ASEN5264/ExData.csv"))

#visualize raw data
# plt_raw = display(plt.plot(data_restricted.datetime, data_restricted.gold_price_usd))
# plt_raw_diff = display(plt.plot(data_restricted.datetime, data_restricted.gold_price_usd_diff))


# #baum-welch attempt
init_guess = [0.5, 0.5]
trans_guess = [0.8 0.2; 0.2 0.8]
dists_guess = [dists.MvNormal([15,0],[5 0; 0 1]), dists.MvNormal([5, 0],[2 0; 0 3])]

hmm_guess = hmms.HMM(init_guess, trans_guess, dists_guess)

#thread observations
obs_seq = Vector{Float64}[]
for i=1:length(data.HR_bl_diff)

    push!(obs_seq, [data.HR_bl_diff[i], data.Rsp_Amp_bl_diff[i]])

end

@show obs_seq


# @show typeof([data.HR_bl_diff, data.Rsp_Amp_bl_diff])

@show hmm_est, llh_evolution = hmms.baum_welch(hmm_guess,obs_seq)

#use viterbi to characterize most likely states with solved model
colors = ["green", "red"]
labels = ["Low", "High"]

best_state_seq, _ = hmms.viterbi(hmm_est,obs_seq)
plt_viterbi = plt.scatter([],[])
pop!(plt_viterbi.series_list)
for i=1:2
    inds = findall(x -> x==i, best_state_seq)
    y = data.HR_bl_diff[inds]
    plt.scatter!(y,color=colors[i],label=labels[i])

end
display(plt_viterbi)

plt.savefig("gold_fig.png")
