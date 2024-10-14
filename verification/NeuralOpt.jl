#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

module NeuralOpt

using JuMP, SparseArrays, Complementarity, Flux

export add_neural_constraints!, get_neural_net, get_inputs, get_outputs,
NonlinearMode,ChainMode,BigMReluMode,IndicatorReluMode,ComplementarityReluMode

abstract type AbstractNeuralNet end
abstract type NeuralNetMode end

mutable struct NeuralEncoding
    variables::Dict   #added variables
    constraints::Dict #added constraints
end
NeuralEncoding() = NeuralEncoding(Dict(),Dict())

# TODO: Net Layers
# IDEA: Parse a generic neural net into user defined JuMP constraints.
# Figure out structure by inspection of input
mutable struct NeuralNet <: AbstractNeuralNet
    w::Dict{Int64,Dict{Int64,Float64}}
    b::Dict{Int64,Float64}
    inputs::Vector{JuMP.VariableRef}
    outputs::Vector{JuMP.VariableRef}
    encoding::NeuralEncoding #the algebraic formulation that relates inputs to outputs
end
NeuralNet(w::Dict,b::Dict) = NeuralNet(w,b,NeuralEncoding())

function NeuralNet(w::Dict,b::Dict,inputs::Vector,outputs::Vector)
    n_inputs = length(inputs)
    n_outputs = length(outputs)
    n_nodes = length(w) + n_inputs - n_outputs
    n_nodes >= n_inputs || error("Number of activation nodes is less than the number of inputs")
    length(b) == n_nodes + n_outputs - n_inputs || error("input and output dimensions do not match neural net dimensions")

    encoding = NeuralEncoding(Dict(),Dict())

    net = NeuralNet(w,b,inputs,outputs,encoding)
    return net
end
_check_for_net!(m::JuMP.Model) = haskey(m.ext,:nets) || (m.ext[:neural_nets] = Vector{NeuralNet}())

#Nonlinear neural net function for each node
struct NonlinearMode <: NeuralNetMode
    activation_func::Function
end

#Just inputs and outputs
struct ChainMode <: NeuralNetMode
    chain::Flux.Chain
end

#Indicator Relu activation function
struct IndicatorReluMode <: NeuralNetMode
end

#Complementarity Relu activation function
struct ComplementarityReluMode <: NeuralNetMode
    formulation::Symbol
    mpec_tol::Float64
end
ComplementarityReluMode() = ComplementarityReluMode(:simple,1e-8)

#Indicator w/ Big M activation function
struct BigMReluMode <: NeuralNetMode
    M::Float64
end
BigMReluMode() = BigMReluMode(1e6)

function add_neural_constraints!(m::JuMP.Model,x::Vector{JuMP.VariableRef},y::Vector{JuMP.VariableRef},w::Dict,b::Dict,mode::NeuralNetMode)
    _check_for_net!(m)
    net = NeuralNet(w,b,x,y)

    add_neural_constraints!(m,net,mode)

    push!(m.ext[:neural_nets],net)
    return net
end

function add_neural_constraints!(m::JuMP.Model,x::Vector{JuMP.VariableRef},y::Vector{JuMP.VariableRef},chain::Flux.Chain,mode::NeuralNetMode;sparsify = false)
    if sparsify == true
        w,b = sparse_chain_to_dict(chain)
    else
        w,b = chain_to_dict(chain)
    end
    net = add_neural_constraints!(m,x,y,w,b,mode)
    return net
end

function sparse_chain_to_dict(chain::Flux.Chain)
    n_inputs = size(chain.layers[1].weight)[2]
    w = Dict()
    b = Dict()
    node = n_inputs + 1
    from_offset = 0

    W_layers = [layer.weight for layer in chain.layers]
    neurons_dead_layers = []

    for l = 1:length(W_layers)
        W = W_layers[l]
        layer = chain.layers[l]

        #Figure out active neurons
        if l < length(W_layers)
            row_sum = vec(sum(W;dims = 2))
            row_dead = findall(row_sum .== 0)     #neurons in this layer that receive no inputs
            col_sum = vec(sum(W_layers[l+1];dims = 1))
            col_dead = findall(col_sum .== 0) #dead neurons in next layer.  Empty for first layer.
        else #don't remove outputs from the dictionary
            row_dead = Int64[]
            col_dead = Int64[]
        end

        neurons_dead = union(row_dead,col_dead)
        n_nodes,n_from = size(W)

        #Neurons active in this layer
        neurons_active = setdiff(1:n_nodes,neurons_dead)
        push!(neurons_dead_layers,neurons_dead)

        #Figure out active inputs
        if l == 1  #don't remove the inputs from the dictionary
            # col_sum_layer = vec(sum(W_layers[l];dims = 1))
            # col_dead_layer = findall(col_sum_layer .== 0)
            # inputs_active = setdiff(1:n_from,col_dead_layer)
            inputs_active = 1:n_from
        else #exclude dead neurons from previous layer
            previous_neurons_dead = neurons_dead_layers[l-1]
            col_sum_layer = vec(sum(W_layers[l];dims = 1))
            #col_dead_layer = findall(col_sum_layer .== 0)
            col_dead_layer = union(findall(col_sum_layer .== 0),previous_neurons_dead)
            inputs_active = setdiff(1:n_from,col_dead_layer)
        end

        n_nodes_active = length(neurons_active)
        n_inputs_active = length(inputs_active)
        for i = 1:n_nodes_active
            w[node] = Dict()
            for j = 1:n_inputs_active
                 w[node][j+from_offset] = Float64(W[neurons_active[i],inputs_active[j]])
                #w[node][weights_active[j]+from_offset] = Float64(W[neurons_active[i],weights_active[j]])
            end
            b[node] = Float64(layer.bias[neurons_active[i]])
            node += 1
        end
        from_offset += n_inputs_active
    end
    return w,b
end

function chain_to_dict(chain::Flux.Chain)
    n_inputs = size(chain.layers[1].W)[2]
    w = Dict()
    b = Dict()
    node = n_inputs + 1
    from_offset = 0
    for layer in chain.layers
        W = layer.W
        n_nodes,n_from = size(W)
        for i = 1:n_nodes
            w[node] = Dict()
            for j = 1:n_from
                w[node][j+from_offset] = Float64(W[i,j])
            end
            b[node] = Float64(layer.b[i])
            node += 1
        end
        from_offset += n_from
    end
    return w,b
end

function add_neural_constraints!(m::JuMP.Model,net::NeuralNet,mode::IndicatorReluMode)
    x = net.inputs
    y = net.outputs
    w = net.w
    b = net.b

    n_inputs = length(x)
    n_outputs = length(y)
    n_nodes = length(net.w) + n_inputs - n_outputs

    inputs = 1:n_inputs
    nodes = 1:n_nodes
    intermediate_nodes = n_inputs+1:n_nodes
    outputs = n_nodes+1:n_nodes+n_outputs

    #map output y to the output indices
    y_map = Dict()
    for (i,idx) in enumerate(outputs)
        y_map[idx] = i
    end

    # pre-activation values
    zhat = @variable(m,[i = intermediate_nodes],base_name="zhat")

    # post-activation values
    z = @variable(m,[i = nodes],base_name="z")
    # activation indicator q=0 means z=zhat (positive part of the hinge)
    # q=1 means we are on the zero part of the hinge
    q = @variable(m,[i = intermediate_nodes],Bin,base_name="q")

    #Constraints
    input_con = @constraint(m,[i = inputs],z[i] == x[i])
    zhat_act_con = @constraint(m,[i = intermediate_nodes], zhat[i] == sum(w[i][j]*z[j] for j in keys(w[i])) + b[i])
    z_lower_con = @constraint(m,[i = intermediate_nodes], z[i] >= 0)
    zhat_upper_con = @constraint(m,[i = intermediate_nodes], z[i] >= zhat[i])
    output_con = @constraint(m,[i = outputs],y[y_map[i]] == sum(w[i][j]*z[j] for j in keys(w[i])) + b[i])

    # Indicator constraints
    @constraint(m,indicator1[i = intermediate_nodes], !q[i] => {z[i] <= zhat[i]})
    @constraint(m,indicator2[i = intermediate_nodes], q[i] => {z[i] <= 0})

    variables = Dict(:zhat => zhat,:z => z,:q => q)
    constraints = Dict(:input => input_con,:zhat_activation => zhat_act_con,:z_lower => z_lower_con,
    :zhat_upper => zhat_upper_con,:output => output_con,:indicator => [indicator1,indicator2])
    net.encoding.variables = variables
    net.encoding.constraints = constraints

    return nothing
end

function add_neural_constraints!(m::JuMP.Model,net::NeuralNet,mode::BigMReluMode)
    x = net.inputs
    y = net.outputs
    w = net.w
    b = net.b
    M = mode.M

    n_inputs = length(x)
    n_outputs = length(y)
    n_nodes = length(net.w) + n_inputs - n_outputs

    #Define sets of indices
    inputs = 1:n_inputs
    nodes = 1:n_nodes
    intermediate_nodes = n_inputs+1:n_nodes
    outputs = n_nodes+1:n_nodes+n_outputs

    #map output y to the output indices
    y_map = Dict()
    for (i,idx) in enumerate(outputs)
        y_map[idx] = i
    end

    zhat = @variable(m,[i = intermediate_nodes],base_name="zhat")
    z = @variable(m,[i = nodes],base_name="z")
    q = @variable(m,[i = intermediate_nodes],Bin,base_name="q")

    #Constraints
    input_con = @constraint(m,[i = inputs],z[i] == x[i])
    zhat_act_con = @constraint(m,[i = intermediate_nodes], zhat[i] == sum(w[i][j]*z[j] for j in keys(w[i])) + b[i])
    z_lower_con = @constraint(m,[i = intermediate_nodes], z[i] >= 0)
    zhat_upper_con = @constraint(m,[i = intermediate_nodes], z[i] >= zhat[i])
    output_con = @constraint(m,[i = outputs],y[y_map[i]] == sum(w[i][j]*z[j] for j in keys(w[i])) + b[i])

    # Activation binary constraints
    zhat_positive_con = @constraint(m,[i = intermediate_nodes],z[i] <= zhat[i] + M*q[i])
    zhat_negative_con = @constraint(m,[i = intermediate_nodes],z[i] <= M*(1.0 - q[i]))

    variables = Dict(:zhat => zhat,:z => z,:q => q)
    constraints = Dict(:input => input_con,:zhat_activation => zhat_act_con,:z_lower => z_lower_con,
    :zhat_upper => zhat_upper_con,:output => output_con,:zhat_positive => zhat_positive_con,:zhat_negative => zhat_negative_con)
    net.encoding.variables = variables
    net.encoding.constraints = constraints

    return nothing
end

function add_neural_constraints!(m::JuMP.Model,net::NeuralNet,mode::ComplementarityReluMode)
    x = net.inputs
    y = net.outputs
    w = net.w
    b = net.b
    form = mode.formulation
    mpec_tol = mode.mpec_tol

    n_inputs = length(x)
    n_outputs = length(y)
    n_nodes = length(net.w) + n_inputs - n_outputs


    inputs = 1:n_inputs
    nodes = 1:n_nodes
    intermediate_nodes = n_inputs+1:n_nodes
    outputs = n_nodes+1:n_nodes+n_outputs

    #map output y to the output indices
    y_map = Dict()
    for (i,idx) in enumerate(outputs)
        y_map[idx] = i
    end

    # pre-activation values
    zhat = @variable(m,[i = intermediate_nodes],base_name="zhat")

    # post-activation values
    z = @variable(m,[i = nodes],base_name="z")

    #Constraints
    input_con = @constraint(m,[i = inputs],z[i] == x[i])
    zhat_act_con = @constraint(m,[i = intermediate_nodes], zhat[i] == sum(w[i][j]*z[j] for j in keys(w[i])) + b[i])
    output_con = @constraint(m,[i = outputs],y[y_map[i]] == sum(w[i][j]*z[j] for j in keys(w[i])) + b[i])

    # complementarity constraints
    comp_constraints = []
    for i = intermediate_nodes
        if form == :simple
            ref = @complements(m, 0 <= (z[i] - zhat[i]), z[i] >= 0,:simple,mpec_tol = mpec_tol)
        elseif form == :smooth
            ref = @complements(m, 0 <= (z[i] - zhat[i]), z[i] >= 0,:smooth,mpec_tol = mpec_tol)
        else
            error("form $form not supported by ComplementarityRelumode")
        end
        push!(comp_constraints,ref)
    end

    variables = Dict(:zhat => zhat,:z => z)
    constraints = Dict(:input => input_con,:zhat_activation => zhat_act_con,:output => output_con,:complementarity => comp_constraints)

    net.encoding.variables = variables
    net.encoding.constraints = constraints
    return nothing
end

function add_neural_constraints!(m::JuMP.Model,net::NeuralNet,mode::NonlinearMode)
    x = net.inputs
    y = net.outputs
    w = net.w
    b = net.b
    activation_func = mode.activation_func

    n_inputs = length(x)
    n_outputs = length(y)
    n_nodes = length(net.w) + n_inputs - n_outputs


    inputs = 1:n_inputs
    nodes = 1:n_nodes
    intermediate_nodes = n_inputs+1:n_nodes
    outputs = n_nodes+1:n_nodes+n_outputs

    #map output y to the output indices
    y_map = Dict()
    for (i,idx) in enumerate(outputs)
        y_map[idx] = i
    end

    # pre-activation values
    zhat = @variable(m,[i = intermediate_nodes],base_name="zhat")

    # post-activation values
    z = @variable(m,[i = nodes],base_name="z")

    #Constraints
    input_con = @constraint(m,[i = inputs],z[i] == x[i])
    zhat_act_con = @constraint(m,[i = intermediate_nodes], zhat[i] == sum(w[i][j]*z[j] for j in keys(w[i])) + b[i])
    output_con = @constraint(m,[i = outputs],y[y_map[i]] == sum(w[i][j]*z[j] for j in keys(w[i])) + b[i])

    # Activation constraints
    # register an activation function here
    activation(x) = activation_func(x)
    register(m, :activation, 1, activation, autodiff=true)
    @NLconstraint(m,activation_constraint[i = intermediate_nodes],z[i] == activation(zhat[i]) )

    variables = Dict(:zhat => zhat,:z => z)
    constraints = Dict(:input => input_con,:zhat_activation => zhat_act_con,:output => output_con,:activation => activation_constraint)

    net.encoding.variables = variables
    net.encoding.constraints = constraints

    return nothing
end

end #module
