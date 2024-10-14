#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

using JSON
using Flux

function read_acpf_json(f::AbstractString)
    #Parse json file
    f = open(f,"r")
    arr = JSON.parse(f) #arr = JSON.parse(f; dicttype=Dict, inttype=Int64)
    close(f)

    #Construct a Flux.Chain
    layers = []
    for json_layer in arr
        W = hcat(json_layer["W"]...)
        b = json_layer["b"]
        n_nodes,n_inputs = size(W)
        if haskey(json_layer,"mask")
            mask = hcat(json_layer["mask"]...)
            W .*= mask
        end

        activation = json_layer["activation"]
        if activation == "leaky_relu"
            activation = relu
        else
            activation = eval(Symbol(json_layer["activation"]))
        end

        #my_W_initialize(output_size,input_size) = W[1:output_size,1:input_size]
        #my_b_initialize(output_size) = b[1:output_size]
        #push!(layers,Dense(n_inputs,n_nodes,activation,init = my_W_initialize, bias = my_b_initialize))
        push!(layers,Dense(W, b, activation))
    end
    chain = Flux.Chain(layers...)
    return chain
end

function write_json_net(chain,filename)
    #We don't support custom layers with json yet
    json_layers = []
    for layer in chain.layers[2:end-1] #ignoring the scaling layers for now
        push!(json_layers,Dict("W" => layer.W,"b" => layer.b,"activation" => string(layer.σ)))
    end

    open(filename,"w") do f
        JSON.print(f,json_layers)
    end
end

#c and s are sorted
function write_mip_solution_for_python(dref,results,filename)

    y = deepcopy(results["output"])

    pref_out_index,qg_out_indices,w_out_indices,c_out_indices,s_out_indices = map_wcs_outputs(dref)
    pref_out = y[pref_out_index]
    qg_out = y[qg_out_indices]
    w_out = y[w_out_indices]
    c_out = y[c_out_indices]
    s_out = y[s_out_indices]

    buspairs,bus_to_branch_map = get_buspairs(dref)
    new_inds = sortperm(buspairs)

    c_out_sorted = c_out[new_inds]
    s_out_sorted = s_out[new_inds]

    #Write a json file for the power models solution
    y[c_out_indices] = c_out_sorted
    y[s_out_indices] = s_out_sorted

    res = Dict("input" => results["input"], "output" => y,"v_p_max" => results["v_p_max"],"v_q_max" => results["v_q_max"])


    json_string = JSON.json(res)
    open(filename,"w") do f
        write(f, json_string)
    end
end

#c and s are not re-ordered
function write_mip_solution_for_julia(dref,results,filename)
    json_string = JSON.json(results)
    open(filename,"w") do f
        write(f, json_string)
    end
end
