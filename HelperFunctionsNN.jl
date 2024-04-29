module HelperFunctionsNN
import CSV
import DataFrames as DF
import Random
import Distributions as dists
import XLSX
import Statistics
import Plots as plt
include("HelperFunctions.jl")
import .HelperFunctions as hf
using StaticArrays



#create input data for Flux
function setup_data_input(data,features::Vector{Tuple{String,Int}},holdout_prop::Float64,flag_diff)
    # generate dataframe if given a list of files
    if isa(data,Vector{String})
        data = hf.merge_data(data,features)
    end

    # predict delta_trust rather than absolute trust
    if flag_diff
        data.Trust_diff = [diff(data.Trust); 0]
        data.Trust = data.Trust_diff
    end

    column_headers = Vector{String}()
    #reconfigure features into correct column names
    for j=1:length(features)
        push!(column_headers,features[j][1]*"_"*string(features[j][2]))
    end

    shuffled_data = data[DF.shuffle(1:DF.nrow(data)),:]
    test_data = shuffled_data[1:Int(floor(holdout_prop*DF.nrow(data))),:]
    train_data = shuffled_data[1+Int(floor(holdout_prop*DF.nrow(data))):DF.nrow(data),:]


    input_data = Vector{Tuple{Vector{Float32},Vector{Float32}}}()
    for i=1:DF.nrow(train_data)
        tmp = Vector{Float32}()
        for j=1:length(column_headers)
            push!(tmp,train_data[i,column_headers[j]])
        end
        push!(input_data,(tmp,[train_data[i,"Trust"]]))
    end

    holdout_data = Vector{Tuple{Vector{Float32},Vector{Float32}}}()
    for i=1:DF.nrow(test_data)
        tmp = Vector{Float32}()
        for j=1:length(column_headers)
            push!(tmp,test_data[i,column_headers[j]])
        end
        push!(holdout_data,(tmp,[test_data[i,"Trust"]]))
    end
    return input_data,holdout_data

end

function get_predictor_data(data::Vector{Tuple{Vector{Float32},Vector{Float32}}})

    predictor_data = Vector{Vector{Float32}}()
    for i=1:length(data)
        push!(predictor_data, data[i][1])
    end

    return predictor_data
end

function get_predictee_data(data::Vector{Tuple{Vector{Float32},Vector{Float32}}})

    predictee_data = Vector{Vector{Float32}}()
    for i=1:length(data)
        push!(predictee_data, data[i][2])
    end

    return predictee_data
end

function visualize_classification_results(m,data)

    #get relevant data into an easier form to plot with
    y_predicted = m.(get_predictor_data(data))
    y_predicted_plot = [y_predicted[j][1] for j in 1:length(y_predicted)]
    y_truth = get_predictee_data(data)
    y_truth_plot = [y_truth[j][1] for j in 1:length(y_truth)]

    #color by whether net correctly predicted high trust or low trust state
    # colors = ifelse.( (y_predicted_plot.>=0.5 .&& y_truth_plot .>= 0.5) .|| (y_predicted_plot .< 0.5 .&& y_truth_plot .< 0.5) , "green","red")
    # labels = ifelse.( (y_predicted_plot.>=0.5 .&& y_truth_plot .>= 0.5) .|| (y_predicted_plot .< 0.5 .&& y_truth_plot .< 0.5) , "Correct Prediction","Incorrect Prediction")

    #color by whether predicted trust within 0.1 of actual trust
    pred_error =  abs.(y_predicted_plot .- y_truth_plot)
    colors = ifelse.( pred_error.<=0.1 , "green","red")
    labels = ifelse.( pred_error.<=0.1 , "Correct","Incorrect")


    x = 1:length(y_truth_plot)
    p = plt.scatter(pred_error,y_truth_plot,color=colors,xlabel="Prediction error", ylabel="Survey Trust")

    display(p)

    return p
end

function cummulative_classification_plot(m,data)

    #get relevant data into an easier form to plot with
    y_predicted = m.(get_predictor_data(data))
    y_predicted_plot = [y_predicted[j][1] for j in 1:length(y_predicted)]
    y_truth = get_predictee_data(data)
    y_truth_plot = [y_truth[j][1] for j in 1:length(y_truth)]

    #color by whether net correctly predicted high trust or low trust state
    # colors = ifelse.( (y_predicted_plot.>=0.5 .&& y_truth_plot .>= 0.5) .|| (y_predicted_plot .< 0.5 .&& y_truth_plot .< 0.5) , "green","red")
    # labels = ifelse.( (y_predicted_plot.>=0.5 .&& y_truth_plot .>= 0.5) .|| (y_predicted_plot .< 0.5 .&& y_truth_plot .< 0.5) , "Correct Prediction","Incorrect Prediction")

    #
    errors = sort(abs.(y_predicted_plot .- y_truth_plot))
    error_cdf = Vector{Float64}()
    x = 0:0.01:1
    for i=1:length(x)

        #count the proprtion of occurrences of errors <= current error
        push!(error_cdf, length(findall( <=(x[i]), errors))/length(errors))

    end

    p = plt.plot(x,error_cdf,xlabel="Trust Prediction Error",ylabel="Cummulative Proportion")

    return p


end





end