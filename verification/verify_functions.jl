#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

include((@__DIR__)*("/utils.jl"))

function calculate_branch_flows_wcs(test_input,test_output,dref)

    pd, qd, pg, wi = test_input
    pr, qg, wo, c, s = test_output

    #Actually, I do need all of the branches
    # g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to = load_buspair_data(dref)
    g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to = load_branch_data(dref)
    bus_pairs,branch_to_buspair = get_buspairs(dref)

    pair_map = Dict()
    for (i,pair) in enumerate(bus_pairs)
        pair_map[pair] = i
    end

    n_buspairs = length(dref[:buspairs])
    n_buses = length(dref[:bus])
    n_branches = length(dref[:branch])

    bus_in_indices,bus_out_indices = get_in_out_bus_indices(dref)

    w = Vector(undef,n_buses)
    w[bus_in_indices] = wi
    w[bus_out_indices] = wo

    w_fr = w[bus_fr]
    w_to = w[bus_to]

    s_branch = Vector(undef,n_branches)
    c_branch = Vector(undef,n_branches)
    for i = 1:n_branches
        buspair = branch_to_buspair[i]
        c_branch[i] = c[pair_map[buspair]]
        s_branch[i] = s[pair_map[buspair]]
    end

    ############################################################################################################################################################################################################################################

    p_fr = Vector(undef,n_branches)
    q_fr = Vector(undef,n_branches)
    p_to = Vector(undef,n_branches)
    q_to = Vector(undef,n_branches)
    #for i = 1:n_buspairs
    for i = 1:n_branches
        p_fr[i] = (g[i]+g_fr[i])/tm[i]*w_fr[i] + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(c_branch[i]) + (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(s_branch[i])
        q_fr[i] = -(b[i]+b_fr[i])/tm[i]*w_fr[i] - (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(c_branch[i]) + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(s_branch[i])
        p_to[i] = (g[i]+g_to[i])*w_to[i] + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(c_branch[i]) +  (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(-s_branch[i])
        q_to[i] = -(b[i]+b_to[i])*w_to[i] - (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(c_branch[i]) + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(-s_branch[i])
    end

    #POWER BALANCE RESIDUALS
    p = [p_fr;p_to]
    q = [q_fr;q_to]
    return p,q
end

function calculate_residuals_wcs(test_input,test_output,dref)

    pd, qd, pg, wi = test_input
    pr, qg, wo, c, s = test_output

    bs,gs = load_shunt_data(dref)
    bus_loads,load_to_bus,bus_gens,gen_to_bus,bus_shunts,shunt_to_bus,bus_arcs,arc_to_bus = load_mappings(dref)
    qd_zero_inds,pg_zero_inds,qd_nonzero_inds,pg_nonzero_inds = get_fixed_gen_load_indices(dref)
    slack_gen_index = get_slack_gen_index(dref)
    bus_in_indices,bus_out_indices = get_in_out_bus_indices(dref)

    n_gens = length(dref[:gen])
    n_loads = length(dref[:load])
    n_buses = length(dref[:bus])

    pg_network = Vector(undef,n_gens)
    qd_network = Vector(undef,n_loads)
    w = Vector(undef,n_buses)
    w[bus_in_indices] = wi
    w[bus_out_indices] = wo

    pg_input_inds = setdiff(pg_nonzero_inds,slack_gen_index)
    pg_network[pg_input_inds] .= pg
    pg_network[pg_zero_inds] .= 0.0
    pg_network[slack_gen_index] = pr

    qd_network[qd_nonzero_inds] .= qd
    qd_network[qd_zero_inds] .= 0.0

    p,q = calculate_branch_flows_wcs(test_input,test_output,dref)

    p_res = Vector(undef,n_buses)
    q_res = Vector(undef,n_buses)
    for i = 1:n_buses
        pgen = Float64[pg_network[g] for g in bus_gens[i]]
        pload = Float64[pd[l] for l in bus_loads[i]]
        gshunt = Float64[gs[s] for s in bus_shunts[i]]

        qgen = Float64[qg[g] for g in bus_gens[i]]
        qload = Float64[qd_network[l] for l in bus_loads[i]]
        bshunt = Float64[bs[s] for s in bus_shunts[i]]

        p_res[i] = sum(p[a] for a in bus_arcs[i]) - (sum(pgen) - sum(pload) - sum(gshunt)*w[i])
        q_res[i] = sum(q[a] for a in bus_arcs[i]) - (sum(qgen) - sum(qload) + sum(bshunt)*w[i])
    end

    return p_res,q_res
end

function convert_to_wcs_in_out(dref,input,solution)

    n_loads = length(dref[:load])
    n_gens = length(dref[:gen])
    n_buses = length(dref[:bus])
    n_buspairs = length(dref[:buspairs])

    pd_in_indices,qd_in_indices,pg_in_indices,w_in_indices =  map_wcs_inputs(dref)
    pref_out_index,qg_out_indices,w_out_indices,c_out_indices,s_out_indices = map_wcs_outputs(dref)
    bus_in_indices,bus_out_indices = get_in_out_bus_indices(dref)
    g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to = load_branch_data(dref)
    qd_zero_inds,pg_zero_inds,qd_nonzero_inds,pg_nonzero_inds = get_fixed_gen_load_indices(dref)
    slack_gen_index = get_slack_gen_index(dref)

    bus_pairs,branch_to_buspair = get_buspairs(dref)

    pd = Float64[input["load"]["$i"]["pd"] for i = 1:n_loads] #active load
    qd = Float64[input["load"]["$i"]["qd"] for i = 1:n_loads] #reactive load

    pg = Vector(undef,n_gens) #active generation
    vm = Vector(undef,n_buses) #generator voltage setpoint converted to w (i.e. w_in)
    va = Vector(undef,n_buses)
    qg = Vector(undef,n_gens)

    #generators
    for i = 1:length(solution["gen"])
        gen_bus = input["gen"]["$i"]["gen_bus"]
        if input["bus"]["$(gen_bus)"]["bus_type"] == 3
            pref_out = solution["gen"]["$i"]["pg"]
            pg[i] = pref_out
        else @assert input["bus"]["$(gen_bus)"]["bus_type"] == 2
            pg_input = input["gen"]["$i"]["pg"]
            pg[i] = pg_input
        end
        qg[i] = solution["gen"]["$i"]["qg"]
    end

    for i = 1:length(solution["bus"])
        vm[i] = solution["bus"]["$i"]["vm"]
        va[i] = solution["bus"]["$i"]["va"]
    end

    # bus_fr_pair = branch_to_buspair(bus_fr)
    # bus_to_pair = branch_to_buspair(bus_to)
    #######################################
    c_out = Vector(undef,n_buspairs)
    s_out = Vector(undef,n_buspairs)
    for i = 1:n_buspairs
        buspair = bus_pairs[i]
        #TODO: Get branch to buspair working
        vm_fr = vm[buspair[1]]
        vm_to = vm[buspair[2]]
        va_fr = va[buspair[1]]
        va_to = va[buspair[2]]

        c_out[i] =  vm_fr*vm_to*cos(va_fr - va_to)
        s_out[i] =  vm_fr*vm_to*sin(va_fr - va_to)
    end
    #Setup vectors to calculate branch flows
    w = vm.^2

    w_in = w[bus_in_indices]
    w_out = w[bus_out_indices]
    qg_out = qg
    pd_in = pd
    qd_in = qd[qd_nonzero_inds]
    pg_in = pg[setdiff(pg_nonzero_inds,slack_gen_index)]
    pr_out = pg[slack_gen_index]


    test_input = [pd_in,qd_in,pg_in,w_in]
    test_output = [pr_out,qg_out,w_out,c_out,s_out]

    return test_input,test_output
end


function calculate_branch_flows_nonlinear(vm,va,branch_data)
    bus_fr,bus_to,g,g_fr,g_to,b,b_fr,b_to,tm,tr,ti = branch_data

    #map bus voltages to branches
    vm_fr = vm[bus_fr,:]
    vm_to = vm[bus_to,:]
    va_fr = va[bus_fr,:]
    va_to = va[bus_to,:]

    #calculate branch flows
    p_fr = (g+g_fr)./tm.*vm_fr.^2 + (-g.*tr+b.*ti)./tm.*(vm_fr.*vm_to.*cos.(va_fr-va_to)) + (-b.*tr-g.*ti)./tm.*(vm_fr.*vm_to.*sin.(va_fr-va_to))
    p_to = (g+g_to).*vm_to.^2 + (-g.*tr.-b.*ti)./tm.*(vm_to.*vm_fr.*cos.(va_to-va_fr)) + (-b.*tr+g.*ti)./tm.*(vm_to.*vm_fr.*sin.(va_to-va_fr))
    q_fr = -(b+b_fr)./tm.*vm_fr.^2 - (-b.*tr-g.*ti)./tm.*(vm_fr.*vm_to.*cos.(va_fr-va_to)) + (-g.*tr+b.*ti)./tm.*(vm_fr.*vm_to.*sin.(va_fr-va_to))
    q_to = -(b+b_to).*vm_to.^2 - (-b.*tr+g.*ti)./tm.*(vm_to.*vm_fr.*cos.(va_fr-va_to)) + (-g.*tr-b.*ti)./tm.*(vm_to.*vm_fr.*sin.(va_to-va_fr))

    #return power and reactive power folws
    p = [p_fr;p_to]
    q = [q_fr;q_to]
    return p,q
end

function calculate_residuals_nonlinear(input,p,q,vm,qg,pref,gs,bs,bus_maps)

    pd,qd,pg_in,vg_set = input


    gen_to_bus,load_to_bus,shunt_to_bus,arc_to_bus = bus_maps

    pg = [pref';pg_in]       #NOTE: This assumes the first generator is the slack bus

    #IDEA: setup vectors of generation and load around each bus using sparse multiplication
    pg_bus = gen_to_bus*pg
    qg_bus = gen_to_bus*qg
    pd_bus = load_to_bus*pd
    qd_bus = load_to_bus*qd
    gs_bus = shunt_to_bus*gs
    bs_bus = shunt_to_bus*bs

    #map branch flows to buses
    p_bus = arc_to_bus*p   #sum of power FROM each bus
    q_bus = arc_to_bus*q

    #power flow from a bus must equal generation minus load minus shunt
    p_res = p_bus - (pg_bus - pd_bus - gs_bus.*vm.^2)
    q_res =  q_bus - (qg_bus - qd_bus + bs_bus.*vm.^2)

    return p_res,q_res
end
