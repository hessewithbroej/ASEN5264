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
using Tables



# features = [("fNIRS_Vers4",71),("fNIRS_Vers4",265)]

#most lasso features for AFP31
# features = [("Features",9),("Features",40),("Features",70),("Features",98),("Features",100),("Features",145),("fNIRS_Vers4",71),("fNIRS_Vers4",265)]

#assorted FNIRs features (all hbo_mus)
# features_neuro = [("fNIRS_Vers4",1),("fNIRS_Vers4",2),("fNIRS_Vers4",3),("fNIRS_Vers4",4),("fNIRS_Vers4",5),("fNIRS_Vers4",6),("fNIRS_Vers4",7),("fNIRS_Vers4",8),("fNIRS_Vers4",9),("fNIRS_Vers4",10),("fNIRS_Vers4",11),("fNIRS_Vers4",12),("fNIRS_Vers4",13),("fNIRS_Vers4",14),("fNIRS_Vers4",15),("fNIRS_Vers4",16),("fNIRS_Vers4",17),("fNIRS_Vers4",18),("fNIRS_Vers4",19),("fNIRS_Vers4",20)]

#AFP30 LASSO'd fnirs features afp 30
features_neuro = [("fNIRS_Vers4",18),("fNIRS_Vers4",20),("fNIRS_Vers4",44),("fNIRS_Vers4",45),("fNIRS_Vers4",49),("fNIRS_Vers4",52),("fNIRS_Vers4",77),("fNIRS_Vers4",96),("fNIRS_Vers4",102),("fNIRS_Vers4",107),("fNIRS_Vers4",121),("fNIRS_Vers4",123),("fNIRS_Vers4",124),("fNIRS_Vers4",128),("fNIRS_Vers4",160),("fNIRS_Vers4",225),("fNIRS_Vers4",240),("fNIRS_Vers4",277),("fNIRS_Vers4",290),("fNIRS_Vers4",322),("fNIRS_Vers4",338),("fNIRS_Vers4",345),("fNIRS_Vers4",348),("fNIRS_Vers4",357)]

#version 4 of all physio features -- "Simple model"
features_physio = [("Features",2),("Features",11),("Features",20),("Features",29), ("Features",38),("Features",47),("Features",56),("Features",65),("Features",74),("Features",83),("Features",92),("Features",101),("Features",110),("Features",119),("Features",128),("Features",137),("Features",146)]


# features = append!(features_neuro,features_physio)
features = features_physio

# data_files = ["C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP28.xlsx"]
data_files = ["C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP28.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP30.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP31.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/FeaturesForModelsAFP32.xlsx"]

data = hf.merge_data(data_files,features)

data = hf.trust_transition_parsing(data,[0,0.5])

println("post merge")

input_data,holdout_data = hfn.setup_data_input(data,"Trust_diff",features,0.20)
input_data_x = hfn.get_predictor_data(input_data)
input_data_y = hfn.get_predictee_data(input_data)
holdout_data_x = hfn.get_predictor_data(holdout_data)
holdout_data_y = hfn.get_predictee_data(holdout_data)

println("Post datasetup")
@show length(features)
m = Chain(Dense(length(features), length(features)*2, leakyrelu), Dense(length(features)*2, length(features)*2, leakyrelu), Dense(length(features)*2, 1))
opt = Flux.setup(ADAM(0.0005), m)

# loss(x, y) = sum((m(x)-y).^2)

function loss(m,x,y)
    return (m(x)[1] - y[1])^2 
end

println("post m")

t = []
learncurve = []
numcurve = []
num_episodes = 50000
best_m = m
best_MSE = 1000000
for i in 1:num_episodes

    Flux.Optimise.train!(loss, m, input_data, opt)


    if i%500 == 0

        println("Episode: $(i)")
        hfn.visualize_classification_results(m,input_data)

        #track learning curv
        tot_MSE = 0
        correct_classifications = 0
        for j=1:length(input_data)
            tot_MSE +=  (m( input_data_x[j] )[1] - input_data_y[j][1])^2
            correct_classifications += Int( abs.(m( input_data_x[j] )[1] - input_data_y[j][1])<=0.1)
        end
        if tot_MSE < best_MSE 
            best_m = m
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

hfn.cummulative_classification_plot(best_m,holdout_data)