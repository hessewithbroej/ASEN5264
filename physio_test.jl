import HiddenMarkovModels as hmms
import CSV
import DataFrames as DF
import Random
import Distributions as dists
import Plots as plt
import Dates
import LogarithmicNumbers as LN


Random.seed!(12321)

#load data, add a differential column
data = DF.DataFrame(CSV.File("C:/Users/hesse/Desktop/Code/ASEN5264/ExData.csv"))

#visualize raw data
# plt_raw = display(plt.plot(data_restricted.datetime, data_restricted.gold_price_usd))
# plt_raw_diff = display(plt.plot(data_restricted.datetime, data_restricted.gold_price_usd_diff))

data.HR_bl_diff = data.HR_bl_diff

# #baum-welch attempt
init_guess = [LN.ULogarithmic(0.0161), LN.ULogarithmic(0.0323), LN.ULogarithmic(0.0645), LN.ULogarithmic(0.1290), LN.ULogarithmic(0.2581), LN.ULogarithmic(0.2581), LN.ULogarithmic(0.1290), LN.ULogarithmic(0.0645), LN.ULogarithmic(0.0323), LN.ULogarithmic(0.0161)]
# init_guess = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
trans_guess = [    
    LN.ULogarithmic(0.34)          LN.ULogarithmic(0.17)          LN.ULogarithmic(0.11)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.07)          LN.ULogarithmic(0.06)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.03);
    LN.ULogarithmic(0.15)          LN.ULogarithmic(0.30)          LN.ULogarithmic(0.15)          LN.ULogarithmic(0.10)          LN.ULogarithmic(0.08)          LN.ULogarithmic(0.06)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.03);
    LN.ULogarithmic(0.09)          LN.ULogarithmic(0.14)          LN.ULogarithmic(0.28)          LN.ULogarithmic(0.14)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.07)          LN.ULogarithmic(0.06)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.04);
    LN.ULogarithmic(0.06)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.14)          LN.ULogarithmic(0.27)          LN.ULogarithmic(0.14)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.07)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.04);
    LN.ULogarithmic(0.05)          LN.ULogarithmic(0.08)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.13)          LN.ULogarithmic(0.27)          LN.ULogarithmic(0.13)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.07)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.04);
    LN.ULogarithmic(0.04)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.08)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.13)          LN.ULogarithmic(0.27)          LN.ULogarithmic(0.13)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.07)          LN.ULogarithmic(0.05);
    LN.ULogarithmic(0.03)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.07)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.14)          LN.ULogarithmic(0.27)          LN.ULogarithmic(0.14)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.07);
    LN.ULogarithmic(0.04)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.06)          LN.ULogarithmic(0.07)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.14)          LN.ULogarithmic(0.28)          LN.ULogarithmic(0.14)          LN.ULogarithmic(0.09);
    LN.ULogarithmic(0.03)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.06)          LN.ULogarithmic(0.08)          LN.ULogarithmic(0.10)          LN.ULogarithmic(0.15)          LN.ULogarithmic(0.30)          LN.ULogarithmic(0.15);
    LN.ULogarithmic(0.03)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.04)          LN.ULogarithmic(0.05)          LN.ULogarithmic(0.06)          LN.ULogarithmic(0.07)          LN.ULogarithmic(0.09)          LN.ULogarithmic(0.11)          LN.ULogarithmic(0.17)          LN.ULogarithmic(0.34);
]

dists_guess = [dists.Normal(15,3), dists.Normal(13,3), dists.Normal(11,3),dists.Normal(9,2),dists.Normal(8,2),dists.Normal(7,2),dists.Normal(6,2),dists.Normal(6,2),dists.Normal(3,1),dists.Normal(0,1)]

hmm_guess = hmms.HMM(init_guess, trans_guess, dists_guess)

@show hmm_est, llh_evolution = hmms.baum_welch(hmm_guess,data.HR_bl_diff)

#use viterbi to characterize most likely states with solved model
# colors = ["green", "yellow", "red"]
# labels = ["Low", "Mid", "High"]

# best_state_seq, _ = hmms.viterbi(hmm_est,data_restricted.gold_price_usd_diff)
# plt_viterbi = plt.scatter([],[])
# pop!(plt_viterbi.series_list)
# for i=1:3
#     inds = findall(x -> x==i, best_state_seq)
#     x = data_restricted.datetime[inds]
#     # @show x = Dates.value.(x-Dates.Date(2008,1,1))
#     y = data_restricted.gold_price_usd_diff[inds]
#     plt.scatter!(x,y,color=colors[i],label=labels[i])

# end
# display(plt_viterbi)

# plt.savefig("gold_fig.png")
