import CSV
import HiddenMarkovModels as hmms
import DataFrames as DF
import Random
import Distributions as dists
import XLSX
include("HelperFunctions.jl")
import .HelperFunctions as hf
include("HelperFunctionsNN.jl")
import .HelperFunctionsNN as hfn
import Plots as plt
using CommonRLInterface
using Flux
using CommonRLInterface.Wrappers: QuickWrapper
using StaticArrays

features = [("fNIRS_Vers4",71),("fNIRS_Vers4",265)]

data_files = ["C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP31.xlsx"]

# data_files = ["C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S1_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S2_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S3_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S4_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S1_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S2_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S3_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S4_Features.xlsx"]


data = hf.merge_data(data_files,features)

data_SA = hfn.setup_data_input(data_files,features)
data_SA_x = hfn.get_predictor_data(data_SA)
data_SA_y = hfn.get_predictee_data(data_SA)


# # data_SA_old = [(SVector(data.fNIRS_Vers4_71[i],data.fNIRS_Vers4_265[i]), SVector(data.Trust[i])) for i in 1:DF.nrow(data)]
# # data_SA_x = [SVector(data.fNIRS_Vers4_71[i],data.fNIRS_Vers4_265[i]) for i in 1:DF.nrow(data)]
# # data_SA_y = [SVector(data.Trust[i]) for i in 1:DF.nrow(data)]

m = Chain(Dense(2, 50, relu), Dense(50, 50, relu), Dense(50, 1))
loss(x, y) = sum((m(x)-y).^2)


t = []
learncurve = []
num_episodes = 50000

for i in 1:num_episodes
    Flux.train!(loss, Flux.params(m), data_SA, Descent(0.05))

    if i%50 == 0

        println("Episode: $(i)")
        hfn.visualize_classification_results(m,data_SA)

        #track learning curv
        tot = 0
        for j=1:length(data_SA)
            tot+=  (m( data_SA_x[j] )[1] - data_SA_y[j][1])^2
        end

        push!(t,i)
        push!(learncurve,  tot )

    end

end

@show learncurve

display(plt.plot(t,learncurve, label="Problem 2 learn curve"))
