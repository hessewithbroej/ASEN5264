import HiddenMarkovModels as hmms
import CSV
import DataFrames as DF
import Random
import Distributions as dists
import Plots as plt
import Dates

Random.seed!(12321)

#load data, add a differential column
data = DF.DataFrame(CSV.File("C:/Users/hesse/Desktop/Code/ASEN5264/gold_price_usd.csv"))
data.gold_price_usd_diff = [0; diff(data.gold_price_usd)]

#data restrict to after to jan 1 2008
after_date(date::Dates.Date) = date > Dates.Date(2008,1,1)
data_restricted = filter(:datetime => after_date, data)

#get data in differential form, supposedly easier to model
# deleteat!(data_restricted, [1])
# @show data_restricted

#visualize raw data
plt_raw = display(plt.plot(data_restricted.datetime, data_restricted.gold_price_usd))
plt_raw_diff = display(plt.plot(data_restricted.datetime, data_restricted.gold_price_usd_diff))



# #baum-welch attempt
init_guess = [0.9, 0.05, 0.05]
trans_guess = [0.8 0.15 0.05; 0.10 0.80 0.10; 0.05 0.1 0.85]
dists_guess = [dists.Normal(1,5), dists.Normal(1,10), dists.Normal(1,20)]

hmm_guess = hmms.HMM(init_guess, trans_guess, dists_guess)
hmm_est, llh_evolution = hmms.baum_welch(hmm_guess,data_restricted.gold_price_usd_diff)

println("done")


#use viterbi to characterize most likely states with solved model

colors = ["green", "yellow", "red"]
labels = ["Low Vol.", "Moderate Vol.", "High Vol."]

best_state_seq, _ = hmms.viterbi(hmm_est,data_restricted.gold_price_usd_diff)
plt_viterbi = plt.scatter([],[], label="")
pop!(plt_viterbi.series_list)
for i=1:3
    inds = findall(x -> x==i, best_state_seq)
    x = data_restricted.datetime[inds]
    # @show x = Dates.value.(x-Dates.Date(2008,1,1))
    y = data_restricted.gold_price_usd_diff[inds]
    plt.scatter!(x,y,xlabel="Date",ylabel="Gold Price Change (USD)",title="Viterbi Classification of Gold Price Volatility", color=colors[i],label=labels[i])

end
display(plt_viterbi)

plt.savefig("gold_fig.png")
