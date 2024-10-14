#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

using LinearAlgebra, SparseArrays

function save_model_solution!(model,data_out)
    pref_out = value.(model[:pref_out])
    pref_model = value.(model[:pg][slack_gen_index])

    qg_out = value.(model[:qg_out])
    qg_model = value.(model[:qg])

    w_out = value.(model[:w_out])
    w_out_model = value.(model[:w][bus_out_indices])

    c_out = value.(model[:c_out])
    c_out_model = value.(model[:c])

    s_out = value.(model[:s_out])
    s_out_model = value.(model[:s])

    push!(data_out,[pref_out,pref_model])
    push!(data_out,[qg_out,qg_model])
    push!(data_out,[w_out,w_out_model])
    push!(data_out,[c_out,c_out_model])
    push!(data_out,[s_out,s_out_model])

    return nothing
end


#Map loads, gens, and shunts to buses
function load_mappings(ref::Dict)
    n_buses = length(ref[:bus])
    n_loads = length(ref[:load])
    n_gens = length(ref[:gen])
    n_shunts = length(ref[:shunt])
    n_branches = length(ref[:branch])
    n_buspairs = length(ref[:buspairs])

    bus_loads = Dict(i => Int64[] for i = 1:n_buses)
    load_to_bus = spzeros(n_buses,n_loads)
    for i = 1:n_loads
        bus = ref[:load][i]["load_bus"]
        push!(bus_loads[bus],i)
        load_to_bus[bus,i] = 1
    end

    bus_gens = Dict(i => Int64[] for i = 1:n_buses)
    gen_to_bus = spzeros(n_buses,n_gens)
    for i = 1:n_gens
        bus = ref[:gen][i]["gen_bus"]
        push!(bus_gens[bus],i)
        gen_to_bus[bus,i] = 1
    end

    bus_shunts = Dict(i => Int64[] for i = 1:n_buses)
    shunt_to_bus = spzeros(n_buses,n_shunts)
    for i = 1:n_shunts
        bus = ref[:shunt][i]["shunt_bus"]
        push!(bus_shunts[bus],i)
        shunt_to_bus[bus,i] = 1
    end

    #Arc mapping
    arc_to_bus = spzeros(n_buses,n_branches*2)
    #bus_pairs_checked = []
    for i = 1:n_branches
        branch = ref[:branch][i]
        fr_bus = branch["f_bus"]
        to_bus = branch["t_bus"]
        #bus_pair = (fr_bus,to_bus)

        # if bus_pair in bus_pairs_checked
        #     continue
        # else
        #     push!(bus_pairs_checked,bus_pair)
        #     j += 1
        # end

        arc_to_bus[fr_bus,i] = 1
        arc_to_bus[to_bus,i+n_branches] = 1 #offset of n_branches needed for reverse arcs
    end

    bus_arcs = Dict{Int64,Vector{Int64}}()
    for i = 1:n_buses
        svec = arc_to_bus[i,:]
        bus_arcs[i] = svec.nzind
    end

    return bus_loads,load_to_bus,bus_gens,gen_to_bus,bus_shunts,shunt_to_bus,bus_arcs,arc_to_bus
end

#load up branch data
function load_branch_data(ref::Dict)
    n_buses = length(ref[:bus])
    n_branches = length(ref[:branch])

    g = [];b = []; tr = [];ti = []
    g_fr = []; b_fr = []; g_to = []; b_to = []; tm = []
    bus_fr = Int64[]
    bus_to = Int64[]
    for i = 1:n_branches
        branch = ref[:branch][i]
        g_calc, b_calc = PowerModels.calc_branch_y(branch)
        tr_calc, ti_calc = PowerModels.calc_branch_t(branch)
        push!(g,g_calc); push!(b,b_calc); push!(tr,tr_calc); push!(ti,ti_calc)
        push!(g_fr,branch["g_fr"]); push!(b_fr,branch["b_fr"]); push!(g_to,branch["g_to"]); push!(b_to,branch["b_to"]); push!(tm,branch["tap"]^2)

        fr_bus = branch["f_bus"]
        to_bus = branch["t_bus"]

        push!(bus_fr,fr_bus)
        push!(bus_to,to_bus)
    end

    return g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to
end

#same as branch data, but only bus pairs
function load_buspair_data(dref::Dict)
    n_buses = length(dref[:bus])
    n_branches = length(dref[:branch])
    n_buspairs = length(dref[:buspairs])

    g = [];b = []; tr = [];ti = []
    g_fr = []; b_fr = []; g_to = []; b_to = []; tm = []
    bus_fr = Int64[]
    bus_to = Int64[]

    bus_pairs_checked = []
    duplicates = []
    j = 0
    for i = 1:n_branches
        branch = dref[:branch][i]
        fr_bus = branch["f_bus"]
        to_bus = branch["t_bus"]
        bus_pair = (fr_bus,to_bus)

        if bus_pair in bus_pairs_checked
            push!(duplicates,i)
            continue
        else
            push!(bus_pairs_checked,bus_pair)
            j += 1
        end

        g_calc, b_calc = PowerModels.calc_branch_y(branch)
        tr_calc, ti_calc = PowerModels.calc_branch_t(branch)
        push!(g,g_calc); push!(b,b_calc); push!(tr,tr_calc); push!(ti,ti_calc)
        push!(g_fr,branch["g_fr"]); push!(b_fr,branch["b_fr"]); push!(g_to,branch["g_to"]); push!(b_to,branch["b_to"]); push!(tm,branch["tap"]^2)

        push!(bus_fr,fr_bus)
        push!(bus_to,to_bus)

    end

    return g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to
end

function get_buspairs(dref::Dict)
    n_branches = length(dref[:branch])
    n_buspairs = length(dref[:buspairs])

    buspairs = Tuple[]
    branch_to_buspair = Dict()

    for i = 1:n_branches
        branch = dref[:branch][i]

        fr_bus = branch["f_bus"]
        to_bus = branch["t_bus"]

        if !((fr_bus,to_bus) in buspairs)
            push!(buspairs,(fr_bus,to_bus))
        end

        branch_to_buspair[i] = (fr_bus,to_bus)
    end
    return buspairs,branch_to_buspair
end

function buspair_to_branch(dref::Dict)
    #need to convert c and s to branches

end

#load shunt data
function load_shunt_data(dref::Dict)
    bs = Float64[]
    gs = Float64[]
    for i = 1:length(dref[:shunt])
        shunt = dref[:shunt][i]
        push!(bs,shunt["bs"])
        push!(gs,shunt["gs"])
    end
    return bs,gs
end

#get the slack bus index and voltage
function load_slack(ref::Dict)
    ref_buses = ref[:ref_buses]
    slack_bus = ref_buses[collect(keys(ref_buses))[1]]
    va_slack = slack_bus["va"]
    slack_index = slack_bus["index"]
    return va_slack,slack_index
end

#Order the input indices for a neural net
function map_net_inputs(ref::Dict)
    n_buses = length(ref[:bus])
    n_gens = length(ref[:gen])
    n_loads = length(ref[:load])

    pgen_buses = 0
    vset_buses = 0

    for i = 1:n_buses
        bus = ref[:bus][i]
        if bus["bus_type"] == 2     #generator buses (PV buses)
            pgen_buses += 1
            vset_buses += 1
        elseif bus["bus_type"] == 3
            vset_buses += 1
        else
            continue
        end
    end

    pload_indices = 1:n_loads
    count = n_loads
    qload_indices = count+1:count+n_loads
    count += n_loads
    pg_indices = count+1:count+pgen_buses
    count += pgen_buses
    vset_indices = count+1:count+vset_buses

    return pload_indices,qload_indices,pg_indices,vset_indices
end

function map_net_outputs(ref::Dict)
    n_buses = length(ref[:bus])
    n_gens = length(ref[:gen])

    vg_set_indices = Int64[]
    vm_predict_indices = Int64[]  #vm for loads
    va_predict_indices = Int64[]  #va for loads and generators, but not slack

    for i = 1:n_buses
        bus = ref[:bus][i]
        if bus["bus_type"] == 1          #load buses (PQ buses)
            push!(vm_predict_indices,i)
            push!(va_predict_indices,i)
        elseif bus["bus_type"] == 2     #generator buses (PV buses)
            push!(va_predict_indices,i)
            push!(vg_set_indices,i)
        else
            @assert bus["bus_type"] == 3
            push!(vg_set_indices,i)
        end
    end

    #vm_map and va_map map the output predictions to bus indices
    vm_map = zeros(Int64,n_buses)
    va_map = zeros(Int64,n_buses)
    for (i,idx) in enumerate(vm_predict_indices)
        vm_map[idx] = i
    end
    vm_offset = length(vm_predict_indices)
    for (i,idx) in enumerate(vg_set_indices)
        vm_map[idx] = i + vm_offset
    end

    for (i,idx) in enumerate(va_predict_indices)
        va_map[idx] = i
    end
    va_slack,slack_index = load_slack(ref)
    va_offset = length(va_predict_indices)
    va_map[slack_index] = va_offset + 1

    count = 1
    vm_out_indices = 1:length(vm_predict_indices)
    count += length(vm_predict_indices)
    va_out_indices = count:count+length(va_predict_indices) - 1
    count += length(va_predict_indices)
    qg_out_indices = count:count + n_gens - 1
    count += n_gens
    pref_out_index = count

    return vm_map,va_map,vm_out_indices,va_out_indices,qg_out_indices,pref_out_index
end

function get_in_out_bus_indices(dref::Dict)
    n_buses = length(dref[:bus])
    bus_in_indices = Int64[]
    bus_out_indices = Int64[]  #vm for loads
    for i = 1:n_buses
        bus = dref[:bus][i]
        if bus["bus_type"] == 1          #load buses (PQ buses)
            push!(bus_out_indices,i)
        else
            @assert bus["bus_type"] == 2 || bus["bus_type"] == 3     #generator buses (PV buses)
            push!(bus_in_indices,i)
        end
    end

    return bus_in_indices,bus_out_indices
end

function get_slack_gen_index(dref::Dict)
    n_gens = length(dref[:gen])
    for i = 1:n_gens
        if dref[:bus][dref[:gen][i]["gen_bus"]]["bus_type"] == 3
            return i
        else
            continue
        end
    end
end

#Find the generator and load indices that are fixed to zero.
function get_fixed_gen_load_indices(dref::Dict)

    n_loads= length(dref[:load])
    n_gens = length(dref[:gen])

    #Get indices that are always set to zero
    qd_data = [dref[:load][i]["qd"] for i = 1:n_loads]
    qd_zero_inds = findall(qd_data .== 0.0)
    pg_data = [dref[:gen][i]["pg"] for i = 1:n_gens]
    pg_zero_inds = findall(pg_data .== 0.0)

    qd_nonzero_inds = setdiff(1:n_loads,qd_zero_inds)
    pg_nonzero_inds = setdiff(1:n_gens,pg_zero_inds)

    return qd_zero_inds,pg_zero_inds,qd_nonzero_inds,pg_nonzero_inds
end

function map_wcs_inputs(dref::Dict)

    n_buses = length(dref[:bus])
    n_gens = length(dref[:gen])
    n_loads = length(dref[:load])

    pgen_buses = 0
    wset_buses = 0

    for i = 1:n_buses
        bus = dref[:bus][i]
        if bus["bus_type"] == 2     #generator buses (PV buses)
            pgen_buses += 1
            wset_buses += 1
        elseif bus["bus_type"] == 3
            wset_buses += 1
        else
            continue
        end
    end

    #Get indices that are always set to zero
    qd_zero_inds,pg_zero_inds,_,_ = get_fixed_gen_load_indices(dref)


    pg_in_indices = 1:pgen_buses - length(pg_zero_inds)
    count = length(pg_in_indices)
    pd_in_indices = count + 1:count+n_loads
    count += length(pd_in_indices)
    qd_in_indices = count+1:count+n_loads - length(qd_zero_inds)
    count += length(qd_in_indices)
    w_in_indices = count+1:count+wset_buses

    return pg_in_indices,pd_in_indices,qd_in_indices,w_in_indices
end

function map_wcs_outputs(dref::Dict)
    n_buses = length(dref[:bus])
    n_gens = length(dref[:gen])
    n_buspairs = length(dref[:buspairs])

    n_bus_output = 0
    for i = 1:n_buses
        bus = dref[:bus][i]
        if bus["bus_type"] == 1          #load buses (PQ buses)
            n_bus_output += 1
        else
            @assert bus["bus_type"] == 2 || bus["bus_type"] == 3     #generator buses (PV buses)
        end
    end

    pref_out_index = 1
    count = 1
    qg_out_indices = count+1:count + n_gens
    count += length(qg_out_indices)
    w_out_indices = count+1:count + n_bus_output
    count += length(w_out_indices)
    c_out_indices = count+1:count + n_buspairs
    count += length(c_out_indices)
    s_out_indices = count + 1:count + n_buspairs

    return pref_out_index,qg_out_indices, w_out_indices,c_out_indices,s_out_indices
end

function load_w_bounds(dref::Dict)
    n_buspairs = length(dref[:buspairs])
    wr_min, wr_max, wi_min, wi_max = PowerModels.ref_calc_voltage_product_bounds(dref[:buspairs])

    g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to = load_branch_data(dref)

    c_lower = zeros(n_buspairs)
    c_upper = zeros(n_buspairs)
    s_lower = zeros(n_buspairs)
    s_upper = zeros(n_buspairs)

    for i = 1:n_buspairs
        c_lower[i] = wr_min[(bus_fr[i],bus_to[i])]
        c_upper[i] = wr_max[(bus_fr[i],bus_to[i])]
        s_lower[i] = wi_min[(bus_fr[i],bus_to[i])]
        s_upper[i] = wi_max[(bus_fr[i],bus_to[i])]
    end

    return c_lower,c_upper,s_lower,s_upper
end




# w_map = zeros(Int64,n_buses)
# for (i,idx) in enumerate(w_predict_indices)
#     w_map[idx] = i
# end
# w_offset = length(w_predict_indices)
# for (i,idx) in enumerate(w_set_indices)
#     w_map[idx] = i + w_offset
# end
