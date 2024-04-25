import CSV
import DataFrames as DF
import Random
import Distributions as dists

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
        for j=1:DF.nrow(data_subset)
            for k=1:length(features)-1           
                fit_data[k,j] = data_subset[j,k]
            end
        end
        push!(obs_dists, dists.fit_mle(dists.MvNormal, fit_data))

    end

    return obs_dists

end


data = DF.DataFrame(CSV.File("C:/Users/hesse/Desktop/Code/ASEN5264/ExData.csv"))
@show data

states = [0.0, 0.5]

transition_matrix = estimate_transitions(states,data)
observation_distributions = estimate_observations(states, [:HR_bl_diff, :Rsp_Amp_bl_diff], data)


# @show y = classify_states([0.0, 0.5], data)

# @show x= estimate_transitions([0.0, 0.5],data)



# estimate_observations([0.0,0.5],[:Trust,:HR_bl_diff],data)
