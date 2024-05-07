module HelperFunctions
import CSV
import DataFrames as DF
import Random
import Distributions as dists
import XLSX
import Statistics
import Plots as plt
using Tables



#get all subject data into one dataframe
function merge_data(workbooks::Vector{String},features::Vector{Tuple{String,Int}})
    #workbooks: string array of absolute filepaths to data xlsxs
    #features: vector of tuples corresponding to features of interest
        # eg features = [("ECG_RMSSD",2)] would get the second column of the sheet "ECG_RMSSD"


    #get the trust data from all the workbooks' first sheet
    labels = Vector{String}()
    trusts = Vector{Float64}()
    for i=1:length(workbooks)

        m = match(r"AFP[0-9]+",workbooks[i])
        
        trust = DF.DataFrame(XLSX.readtable(workbooks[i],1)).Trust

        append!(trusts,trust)
        tmp = repeat([m.match*"S$(1)"], Int(length(trust)/4))
        append!(labels,tmp)
        tmp = repeat([m.match*"S$(2)"], Int(length(trust)/4))
        append!(labels,tmp)
        tmp = repeat([m.match*"S$(3)"], Int(length(trust)/4))
        append!(labels,tmp)
        tmp = repeat([m.match*"S$(4)"], Int(length(trust)/4))
        append!(labels,tmp)
    end
    #construct initial dataframe with session Ids and trust values. 
    data = DF.DataFrame(SessionID=labels,Trust=trusts)

    #get specified features from all the workbooks and add them to dataframe 
    tmp_data = []
    for j=1:length(features)

        #aggregate current feature's data from all workbooks
        tmp_data = Vector{Any}()
        for i=1:length(workbooks)

            append!(tmp_data,DF.DataFrame(XLSX.readtable(workbooks[i],features[j][1]))[:,features[j][2]])
        
        end
        data.tmp = tmp_data
        DF.rename!(data,:tmp => features[j][1]*"_"*string(features[j][2]) )

    end

    return data

end

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

# parse trust data and generate a column of "transitions" ie diff between curr trust state and nex
function trust_transition_parsing(data::DF.DataFrame,states)


    #ensure dataframe already has stateindex 
    try
        data.StateIndex
    catch
        # if isa(exc, ArgumentError)
            data = classify_states(states,data)
        # else
            # error(exc)
        # end
    end
    
    # data.StateTransition = [0; diff(data.StateIndex)]
    data.StateTransition = [data.StateIndex[2:length(data.StateIndex)]; 0]




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

    #additional validation - check if multiple sessionIDs or not
    flag_multiple_sesisons = true
    try 
        data.SessionID
    catch exc
        if isa(exc, ArgumentError)
            println("No SessionID found in dataframe. Assuming all data comes from one session")
            flag_multiple_sesisons = false
        else
            error(exc)
        end
    end

    transitions = zeros(Float64,length(states),length(states))

    for i=1:DF.nrow(data)-1
        
        #make sure we're not including inter-session transitions
        curr_session = data[i,:SessionID]
        next_session = data[i+1,:SessionID]
        if curr_session != next_session
            continue
        else 
            #get state of current timestep
            curr_state = data[i,:StateIndex]

            #get state of next timestep
            # next_trust = data[i+1,1]
            # tmp = findall(<=(next_trust), states)
            next_state =  data[i+1,:StateIndex]
            
            #update transitions Matrix
            transitions[curr_state,next_state] += 1
        end

    end

    #normalize rows
    transitions = transitions ./ sum(transitions,dims=2)
    # #weird numerical imprecision bug, temporary workaround by rrounding then fixing the last entry
    # transitions = round.(transitions,digits=4)
    # for j=1:length(states)
    #     transitions[j,length(states)] = round(1-sum(transitions[j,1:length(states)-1]),digits=4)
    # end

    return transitions

end


#generate observation distributions for each specifed state using the specified features/physio data
function estimate_observations(states::Vector{Float64},features::Vector{Tuple{String,Int}},data::DF.DataFrame,flag_covariation::Bool)::Vector{dists.FullNormal}
    
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

    column_headers = ["StateIndex"]
    #reconfigure features into correct column names
    for j=1:length(features)
        push!(column_headers,features[j][1]*"_"*string(features[j][2]))
    end

    #to be returned
    obs_dists = Vector{dists.FullNormal}()
    if flag_covariation
        #generate a MvNormal distribution object for each trust state
        for i=1:length(states)
            
            #subset data corresponding to this state
            data_subset = data[data.StateIndex .== i, column_headers]
            fit_data = zeros(Float64,length(column_headers)-1,DF.nrow(data_subset))

            #shitty way to collect all the data into the form needed by fit_mle, but it works for now
            for j=1:DF.nrow(data_subset)
                for k=2:length(column_headers)           
                    fit_data[k-1,j] = data_subset[j,k]
                end
            end

            #use the Distributions.fit_mle command to execute the multivariate gaussian fit algorithm, save its results.
            push!(obs_dists, dists.fit_mle(dists.MvNormal, fit_data))
        end

    #disallow covariance in the distributions
    else
        for i=1:length(states)
            
            #subset data corresponding to this state
            data_subset = data[data.StateIndex .== i, column_headers]
            fit_data = zeros(Float64,length(column_headers)-1,DF.nrow(data_subset))

            #shitty way to collect all the data into the form needed by fit_mle, but it works for now
            for j=1:DF.nrow(data_subset)
                for k=2:length(column_headers)           
                    fit_data[k-1,j] = data_subset[j,k]
                end
            end

            mu = Vector{Float64}()
            sigma = zeros(Float64,length(column_headers)-1,length(column_headers)-1)
            #compute the univariate mean and stddev of each observation feature
            for k=1:length(column_headers)-1
                push!(mu,Statistics.mean(fit_data[k,:]))
                sigma[k,k] = Statistics.std(fit_data[k,:])
            end
            
            #use the Distributions.fit_mle command to execute the multivariate gaussian fit algorithm, save its results.
            push!(obs_dists, dists.MvNormal(mu,sigma))
        end


    end


    return obs_dists

end


function create_estimates(states::Vector{Float64},features::Vector{Tuple{String,Int}},data_files::Vector{String})

    data = merge_data(data_files,features)

    #generate the transition matrix estimate based on the provided trust discretization
    transition_matrix = estimate_transitions(states,data)

    # #generate uni/multivariate gaussian observation distributions for each state using the specifed feature(s)
    observation_distributions = estimate_observations(states, features, data, false)

    return (transition_matrix,observation_distributions)

end

# Convert observation data into sets of observation by timestep 
# eg obs_seq = [[var1_time1, var2_time2], [var1_time2, var2_time2], ... []]
function thread_observations(data,features::Vector{Tuple{String,Int}})

    # generate dataframe if given a list of files
    if isa(data,Vector{String})
        data = merge_data(data,features)
    end

    column_headers = Vector{String}()
    #reconfigure features into correct column names
    for j=1:length(features)
        push!(column_headers,features[j][1]*"_"*string(features[j][2]))
    end

    studies = unique(data.SessionID)
    #thread observations for use in HMM.baum_welch. Note where new sequences begin.
    obs_seq = Vector{Vector{Float64}}()
    seq_ends = Vector{Int64}()
    ind = 0
    for k=1:length(studies)
        data_subset = data[data.SessionID .== studies[k],:]
        for i=1:DF.nrow(data_subset)
            ind += 1
            tmp = Vector{Any}()
            for j=1:length(column_headers)
                col = column_headers[j]
                push!(tmp,data_subset[i,col])
            end
            push!(obs_seq, tmp)
        end
        push!(seq_ends,ind)
    end

    return obs_seq,seq_ends

end


function plot_observation_distributions(states::Vector{Float64},features::Vector{Tuple{String,Int}},data::DF.DataFrame)
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
    
    column_headers = Vector{String}()
    #reconfigure features into correct column names
    for j=1:length(features)
        push!(column_headers,features[j][1]*"_"*string(features[j][2]))
    end

    obs_dists = estimate_observations(states,features,data,true)

    colors = ["red","green","yellow","orange"]

    plot = plt.scatter([],[])
    pop!(plot.series_list)
    for i=1:length(states)

        #subset data, just the ccurrent state
        data_subset = data[data.StateIndex.==i,:]

        #extract the two features to be plotted
        #get the data in vectors
        x = data_subset[:,column_headers[1]]
        y = data_subset[:,column_headers[2]]

        plt.scatter!(x,y,color=colors[i],markershape=:+,legend=false)

        #overlay fits
        plt.scatter!([Statistics.mean(x)],[Statistics.mean(y)],color=colors[i],markershape=:star5, markersize=10,legend=false)

    end
    display(plot)
end

end
