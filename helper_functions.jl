import CSV
import DataFrames as DF
import Random
import Distributions as dists
import XLSX

#get all subject data into one dataframe
function merge_data(workbooks::Vector{String},features::Vector{Tuple{String,Int}})
    #workbooks: string array of absolute filepaths to data xlsxs
    #features: vector of tuples corresponding to features of interest
        # eg features = [("ECG_RMSSD",2)] would get the second column of the sheet "ECG_RMSSD"


    #get the trust data from all the workbooks' first sheet
    labels = Vector{String}()
    trusts = Vector{Float64}()
    for i=1:length(workbooks)

        trust = DF.DataFrame(XLSX.readtable(workbooks[1],1)).Trust
        @show trust
        append!(trusts,trust)
        tmp = repeat(["S$(i)"], length(trust))
        append!(labels,tmp)

    end
    #construct initial dataframe with session Ids and trust values. 
    data = DF.DataFrame(SessionID=labels,Trust=trusts)

    #get specified features from all the workbooks and add them to dataframe 
    tmp_data = []
    for j=1:length(features)

        tmp_data = Vector{Any}()
        for i=1:length(workbooks)


            # @show xf = XLSX.readxlsx(workbooks[i])
            # @show column = xf[features[j][1]][:,features[j][2]]

            append!(tmp_data,DF.DataFrame(XLSX.readtable(workbooks[i],features[j][1]))[:,features[j][2]])
            
            
        end
        data.tmp = tmp_data
        DF.rename!(data,:tmp => features[j][1]*"_"*string(features[j][2]) )
        

    end

   
    return data

end


xx = merge_data(["C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S1_Features.xlsx","C:/Users/hesse/Desktop/Code/ASEN5264/AFP31/AFP31_S2_Features.xlsx"], [("ECG_RMSSD",2)])
@show xx



#categorize data into user-specified states based on "Trust" variable
function classify_states(states::Vector{Float64},data::DF.DataFrame)::DF.DataFrame

    #validate states input
    if sort(states) != states
        throw(ArgumentError("states must be listed in increasing order"))
    end
    
    for k = 2:length(states)
        if states[k-1] == states[k]
            throw(ArgumentError("duplicate states found. All states must be unique"))
        end
    end

    state_classification = Vector{Int64}();
    for i=1:DF.nrow(data)
        curr_trust = data[i,:Trust]
        tmp = findall(<=(curr_trust), states)
        curr_state = tmp[length(tmp)]

        push!(state_classification,curr_state)

    end

    data.StateIndex = state_classification
    return data

end


# estimate transition probabilities from ground truth state data
function estimate_transitions(states::Vector{Float64},data::DF.DataFrame)::Matrix{Float64}

    #validate states input
    if sort(states) != states
        throw(ArgumentError("states must be listed in increasing order"))
    end
    
    for k = 2:length(states)
        if states[k-1] == states[k]
            throw(ArgumentError("duplicate states found. All states must be unique"))
        end
    end
    
    #basic input validation
    try
        data.StateIndex
    catch exc
        if isa(exc, ArgumentError)
            data = classify_states(states,data)
        else
            error(exc)
        end
    end

    transitions = zeros(Float64,length(states),length(states))

    for i=1:DF.nrow(data)-1

        #get state of current timestep
        curr_state = data[i,:StateIndex]

        #get state of next timestep
        # next_trust = data[i+1,1]
        # tmp = findall(<=(next_trust), states)
        next_state =  data[i+1,:StateIndex]
        
        #update transitions Matrix
        transitions[curr_state,next_state] += 1

    end

    return transitions*(1/sum(transitions))

end

#generate observation distributions for each specifed state using the specified features/physio data
function estimate_observations(states::Vector{Float64},features::Vector{Symbol},data::DF.DataFrame)::Vector{dists.FullNormal}
    
    #validate states input
    if sort(states) != states
        throw(ArgumentError("states must be listed in increasing order"))
    end
    
    for k = 2:length(states)
        if states[k-1] == states[k]
            throw(ArgumentError("duplicate states found. All states must be unique"))
        end
    end
    
    #basic input validation
    try
        data.StateIndex
    catch exc
        if isa(exc, ArgumentError)
            data = classify_states(states,data)
        else
            error(exc)
        end
    end

    if !(:StateIndex in features)
        push!(features,:StateIndex)
    end

    #to be returned
    obs_dists = Vector{dists.FullNormal}()

    #generate a MvNormal distribution object for each trust state
    for i=1:length(states)
        
        #subset data corresponding to this state
        data_subset = data[data.StateIndex .== i, features]

        fit_data = zeros(Float64,length(features)-1,DF.nrow(data_subset))

        #shitty way to collect all the data into the form needed by fit_mle, but it works for now
        for j=1:DF.nrow(data_subset)
            for k=1:length(features)-1           
                fit_data[k,j] = data_subset[j,k]
            end
        end

        #use the Distributions.fit_mle command to execute the multivariate gaussian fit algorithm, save its results.
        push!(obs_dists, dists.fit_mle(dists.MvNormal, fit_data))

    end

    return obs_dists

end

#some example data
# data = DF.DataFrame(CSV.File("C:/Users/hesse/Desktop/Code/ASEN5264/ExData.csv"))
# @show data

# #trust thresholds for new states. In this example, there are two states, one where 0 <= trust < 0.5, and one where 0.5 <= trust.
# states = [0.0, 0.5]

# #these should be the a list of symbols matching the column headers in whatever excel sheet you're reading from
# features = [:HR_bl_diff, :Rsp_Amp_bl_diff]

# #generate the transition matrix estimate based on the provided trust discretization
# transition_matrix = estimate_transitions(states,data)

# #generate uni/multivariate gaussian observation distributions for each state using the specifed feature(s)
# observation_distributions = estimate_observations(states, features, data)
