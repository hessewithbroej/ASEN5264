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

# features = [("fNIRS_Vers4",71),("fNIRS_Vers4",265)]

#most lasso features for AFP31
# features = [("Features",9),("Features",40),("Features",70),("Features",98),("Features",100),("Features",145),("fNIRS_Vers4",71),("fNIRS_Vers4",265)]

#assorted FNIRs features (all hbo_mus)
features = [("fNIRS_Vers4",1),("fNIRS_Vers4",2),("fNIRS_Vers4",3),("fNIRS_Vers4",4),("fNIRS_Vers4",5),("fNIRS_Vers4",6),("fNIRS_Vers4",7),("fNIRS_Vers4",8),("fNIRS_Vers4",9),("fNIRS_Vers4",10),("fNIRS_Vers4",11),("fNIRS_Vers4",12),("fNIRS_Vers4",13),("fNIRS_Vers4",14),("fNIRS_Vers4",15),("fNIRS_Vers4",16),("fNIRS_Vers4",17),("fNIRS_Vers4",18),("fNIRS_Vers4",19),("fNIRS_Vers4",20)]



data_files = ["C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP28.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP30.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP31.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP32.xlsx"]

# data_files = ["C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S1_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S2_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S3_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP30/AFP30_S4_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S1_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S2_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S3_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S4_Features.xlsx"]


data = hf.merge_data(data_files,features)

println("post merge")

input_data,holdout_data = hfn.setup_data_input(data_files,features,0.15)
input_data_x = hfn.get_predictor_data(input_data)
input_data_y = hfn.get_predictee_data(input_data)
holdout_data_x = hfn.get_predictor_data(holdout_data)
holdout_data_y = hfn.get_predictee_data(holdout_data)

println("Post datasetup")

m = Chain(Dense(length(features), 50, relu), Dense(50, 50, relu), Dense(50, 1))
loss(x, y) = sum((m(x)-y).^2)

println("post m")

t = []
learncurve = []
numcurve = []
num_episodes = 50000

for i in 1:num_episodes
    Flux.train!(loss, Flux.params(m), input_data, Descent(0.05))

    if i%100 == 0

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
global test_ICs20 = 0
for j=1:length(holdout_data)
    global test_MSE +=  (m( holdout_data_x[j] )[1] - holdout_data_y[j][1])^2
    global test_ICs += Int( abs.(m( holdout_data_x[j] )[1] - holdout_data_y[j][1]) <=0.1)
    global test_ICs20 += Int( abs.(m( holdout_data_x[j] )[1] - holdout_data_y[j][1]) <=0.2)
end

@show test_MSE
@show test_ICs

hfn.cummulative_classification_plot(m,holdout_data)