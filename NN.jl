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

input_data,holdout_data = hfn.setup_data_input(data_files,features,0.15)
input_data_x = hfn.get_predictor_data(input_data)
input_data_y = hfn.get_predictee_data(input_data)
holdout_data_x = hfn.get_predictor_data(holdout_data)
holdout_data_y = hfn.get_predictee_data(holdout_data)


m = Chain(Dense(2, 50, relu), Dense(50, 50, relu), Dense(50, 1))
loss(x, y) = sum((m(x)-y).^2)


t = []
learncurve = []
numcurve = []
num_episodes = 100000

for i in 1:num_episodes
    Flux.train!(loss, Flux.params(m), input_data, Descent(0.05))

    if i%50 == 0

        println("Episode: $(i)")
        hfn.visualize_classification_results(m,input_data)

        #track learning curv
        tot_MSE = 0
        correct_classifications = 0
        for j=1:length(input_data)
            tot_MSE +=  (m( input_data_x[j] )[1] - input_data_y[j][1])^2
            correct_classifications += Int( abs.(m( input_data_x[j] )[1] - input_data_y[j][1])<=0.1)
        end

        push!(t,i)
        push!(learncurve,  tot_MSE )
        push!(numcurve,  correct_classifications )

    end

end


display(plt.plot(t,learncurve, label="MSE vs Episode"))
display(plt.plot(t,numcurve, label="Correct Classifications vs Episode"))



hfn.visualize_classification_results(m,holdout_data)

global test_MSE = 0
global test_ICs = 0
for j=1:length(holdout_data)
    global test_MSE +=  (m( holdout_data_x[j] )[1] - holdout_data_y[j][1])^2
    global test_ICs += Int( abs.(m( holdout_data_x[j] )[1] - holdout_data_y[j][1]) <=0.1)
end

@show test_MSE
@show test_ICs