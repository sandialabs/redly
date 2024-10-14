#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#this file defines different optimization-based verification models
#it uses NeuralOpt.jl (in the directory neuralopt) to formulate the neural network models in JuMP

using PowerModelsAnnex, PowerModels
using JuMP

include((@__DIR__)*("/../NeuralOpt.jl"))
include((@__DIR__)*("/../utils.jl"))

#Create a relaxation-based verifier of the AC physics. This verifier can achieve global verification solutions using Gurobi
function build_acpf_wcs_verifier(dref::Dict,chain;nn_mode = NeuralOpt.BigMReluMode(1e6),
    pg_delta = 0.1,pd_delta = 0.1,w_delta = 0.1,qd_delta = 0.1,qd_min = nothing,qd_max = nothing,
    qd_limits = true,
    load_power_factor = :constant,
    gen_power_factor_model = :minimum,
    bound_slack_gen_model = false,
    bound_qg_model = false,
    bound_output_bus_voltage_model = false,
    bound_c_s_model = false,
    thermal_limits_model = false,
    voltage_angle_limits_model = false,
    cut_voltage_angle_limits_model = false,
    gen_power_factor_nn = :off,
    bound_slack_gen_nn = false,
    bound_qg_nn = false,
    bound_output_bus_voltage_nn = false,
    bound_c_s_nn = false,
    thermal_limits_nn = false,
    voltage_angle_limits_nn = false,
    cut_voltage_angle_limits_nn = false,
    soc_nn = false)

    n_buses = length(dref[:bus])
    n_gens = length(dref[:gen])
    n_loads = length(dref[:load])
    n_shunts = length(dref[:shunt])
    n_buspairs = length(dref[:buspairs])
    n_branches = length(dref[:branch])

    #SHUNTS
    bs,gs = load_shunt_data(dref)

    #BRANCH DATA
    g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to = load_branch_data(dref)
    c_lower,c_upper,s_lower,s_upper = load_w_bounds(dref)
    bus_loads,load_to_bus,bus_gens,gen_to_bus,bus_shunts,shunt_to_bus,bus_arcs,arc_to_bus = load_mappings(dref)

    #WCS INPUT AND OUTPUT INDICES for ACPF EQUATIONS
    pg_in_indices,qd_in_indices,pd_in_indices,w_in_indices =  map_wcs_inputs(dref)
    pref_out_index,qg_out_indices,w_out_indices,c_out_indices,s_out_indices = map_wcs_outputs(dref) # w_set are voltage setpoints.  w_out for each bus are the outputs
    ########################################
    #Check input and output dimensions
    ########################################
    input_layer = chain.layers[1]
    output_layer = chain.layers[end]
    n_inputs = size(input_layer.weight)[2]
    n_outputs = size(output_layer.weight)[1]

    @assert n_inputs == length(pd_in_indices) + length(qd_in_indices) + length(pg_in_indices) + length(w_in_indices)
    @assert n_outputs == length(w_out_indices) + length(c_out_indices) + length(s_out_indices) + length(qg_out_indices) + 1 #The 1 is pref generator

    #########################################
    # Model Topology-Based Indices
    #########################################
    slack_gen_index = get_slack_gen_index(dref)
    qd_zero_inds,pg_zero_inds,qd_nonzero_inds,pg_nonzero_inds = get_fixed_gen_load_indices(dref)
    bus_in_indices,bus_out_indices = get_in_out_bus_indices(dref)
    bus_pairs,branch_to_buspair = get_buspairs(dref)
    bus_pairs = sort(bus_pairs)

    #LOAD POWER FACTORS IN MATPOWER
    pd_data = [dref[:load][i]["pd"] for i = 1:n_loads]
    qd_data = [dref[:load][i]["qd"] for i = 1:n_loads]
    S_data = sqrt.(pd_data.^2 + qd_data.^2)
    pf = pd_data ./ S_data
    pf_coeff = sqrt.((1.0 .- pf.^2)./pf.^2)

    #########################################
    # Create Model
    #########################################
    model = Model()

    #VOLTAGES
    @variable(model, w[i in 1:n_buses],start = 1.001)
    for i in bus_out_indices
        JuMP.set_lower_bound(w[i],0)
    end
    @variable(model, c[i in 1:n_buspairs], start=1.0)
    @variable(model, s[i in 1:n_buspairs], start=0.01)

    #GENERATORS
    @variable(model, pg[i in 1:n_gens])
    for i in pg_zero_inds
        JuMP.fix(pg[i],0;force = true) #synchrophasors don't generate active power
    end
    @variable(model, qg[i in 1:n_gens])

    #LOADS
    @variable(model, pd[i in 1:n_loads])
    @variable(model, qd[i in 1:n_loads])
    for i in qd_zero_inds
        JuMP.fix(qd[i],0;force = true)
    end

    ###############################################
    #AC RELAXED BRANCH FLOWS
    ###############################################
    w_fr_bp = [w[bus_pairs[i][1]] for i = 1:n_buspairs]
    w_to_bp = [w[bus_pairs[i][2]] for i = 1:n_buspairs]
    w_fr = w[bus_fr]
    w_to = w[bus_to]

    #MAP BETWEEN BRANCHES AND BUSPAIRS
    pair_map = Dict()
    for (i,pair) in enumerate(bus_pairs)
        pair_map[pair] = i
    end
    s_branch = Vector(undef,n_branches)
    c_branch = Vector(undef,n_branches)
    for i = 1:n_branches
        buspair = branch_to_buspair[i]
        c_branch[i] = c[pair_map[buspair]]
        s_branch[i] = s[pair_map[buspair]]
    end

    ############################################################################################################################################################################################################################################
    @expression(model,p_fr[i=1:n_branches],(g[i]+g_fr[i])/tm[i]*w_fr[i] + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(c_branch[i]) + (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(s_branch[i]))
    @expression(model,q_fr[i=1:n_branches],-(b[i]+b_fr[i])/tm[i]*w_fr[i] - (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(c_branch[i]) + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(s_branch[i]))
    @expression(model,p_to[i=1:n_branches],(g[i]+g_to[i])*w_to[i] + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(c_branch[i]) + (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(-s_branch[i]))
    @expression(model,q_to[i=1:n_branches],-(b[i]+b_to[i])*w_to[i] - (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(c_branch[i]) + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(-s_branch[i]))

    #MODEL POWER BALANCE
    p = [p_fr;p_to]
    q = [q_fr;q_to]
    @constraint(model,active_power[i=1:n_buses],sum(p[a] for a in bus_arcs[i])  == sum(pg[g] for g in bus_gens[i]) - sum(pd[l] for l in bus_loads[i]) - sum(gs[s] for s in bus_shunts[i])*w[i])
    @constraint(model,reactive_power[i=1:n_buses],sum(q[a] for a in bus_arcs[i])  == sum(qg[g] for g in bus_gens[i]) - sum(qd[l] for l in bus_loads[i]) + sum(bs[s] for s in bus_shunts[i])*w[i])

    #########################################
    # NOW ADD VERIFIER CONSTRAINTS
    #########################################
    #SOC constraint on physics (Jabr's relaxation)
    @constraint(model,soc[i=1:n_buspairs], c[i]^2 + s[i]^2 <= w_fr_bp[i]*w_to_bp[i])

    ###############################################
    # INPUT CONSTRAINTS
    ###############################################
    for i = 1:n_loads
        JuMP.set_lower_bound(pd[i],dref[:load][i]["pd"]*(1-pd_delta))
        JuMP.set_upper_bound(pd[i],dref[:load][i]["pd"]*(1+pd_delta))
    end
    for i in pg_nonzero_inds
        if i != slack_gen_index
            JuMP.set_lower_bound(pg[i],dref[:gen][i]["pg"]*(1-pg_delta))
            JuMP.set_upper_bound(pg[i],dref[:gen][i]["pg"]*(1+pg_delta))
        end
    end
    for i in bus_in_indices
        JuMP.set_lower_bound(w[i],dref[:bus][i]["vm"]^2*(1-w_delta))
        JuMP.set_upper_bound(w[i],dref[:bus][i]["vm"]^2*(1+w_delta))
    end

    if qd_limits == true
        if qd_delta != :off
            for i in qd_nonzero_inds
                JuMP.set_lower_bound(qd[i],dref[:load][i]["qd"]*(1-qd_delta))
                JuMP.set_upper_bound(qd[i],dref[:load][i]["qd"]*(1+qd_delta))
            end
        else
            for i in qd_nonzero_inds
                if qd_min != nothing
                    JuMP.set_lower_bound(qd[i],qd_min[i])
                end
                if qd_max != nothing
                    JuMP.set_upper_bound(qd[i],qd_max[i])
                end
            end
        end
    end

    ###############################################
    # MODEL POWER FACTOR CONSTRAINTS
    ###############################################
    #Gen power factor
    if gen_power_factor_model == :constant  #Assume perfect generators
        @constraint(model,pfcongen[i in pg_nonzero_inds],model[:pg][i]^2 == (model[:pg][i]^2 + model[:qg][i]^2))
    elseif gen_power_factor_model == :minimum
        @constraint(model,pfcongen[i in pg_nonzero_inds],model[:pg][i]^2 >= 0.95^2*(model[:pg][i]^2 + model[:qg][i]^2))
    end

    if load_power_factor == :constant
        @constraint(model,pfconload[i in qd_nonzero_inds],model[:qd][i] == pf_coeff[i]*model[:pd][i]) #NOTE: assumes positve reactive load
    elseif load_power_factor == :minimum
        #These are both the same
        #@constraint(model,pfconload[i in qd_nonzero_inds],model[:pd][i]^2 >= pf[i]^2*(model[:pd][i]^2 + model[:qd][i]^2))
        @constraint(model,pfconload[i in qd_nonzero_inds],model[:qd][i]^2 <= (1 - pf[i]^2) / pf[i]^2*(model[:pd][i]^2))
    end

    ###############################################
    #MODEL DEVICE LIMITS
    ###############################################
    if bound_slack_gen_model
        JuMP.set_lower_bound(pg[slack_gen_index],dref[:gen][slack_gen_index]["pmin"])
        JuMP.set_upper_bound(pg[slack_gen_index],dref[:gen][slack_gen_index]["pmax"])
    end

    if bound_qg_model
        for i = 1:n_gens
            JuMP.set_lower_bound(qg[i],dref[:gen][i]["qmin"])
            JuMP.set_upper_bound(qg[i],dref[:gen][i]["qmax"])
        end
    end

    if bound_c_s_model
        for i = 1:n_buspairs
            JuMP.set_lower_bound(c[i],c_lower[i])
            JuMP.set_upper_bound(c[i],c_upper[i])
            JuMP.set_lower_bound(s[i],s_lower[i])
            JuMP.set_upper_bound(s[i],s_upper[i])
        end
    end

    if bound_output_bus_voltage_model
        for i in bus_out_indices
            JuMP.set_lower_bound(w[i],dref[:bus][i]["vmin"]^2)
            JuMP.set_upper_bound(w[i],dref[:bus][i]["vmax"]^2)
        end
    end

    ###############################################
    #MODEL NETWORK CONSTRAINTS
    ###############################################
    if thermal_limits_model
        @constraint(model,thermal_fr[i=1:n_branches],p_fr[i]^2 + q_fr[i]^2 <= dref[:branch][i]["rate_a"]^2)
        @constraint(model,thermal_to[i=1:n_branches],p_to[i]^2 + q_to[i]^2 <= dref[:branch][i]["rate_a"]^2)
    end

    if voltage_angle_limits_model
        angmin = [dref[:buspairs][bus_pairs[i]]["angmin"] for i = 1:n_buspairs]
        angmax = [dref[:buspairs][bus_pairs[i]]["angmax"] for i = 1:n_buspairs]
        @constraint(model,va_diff_min[i=1:n_buspairs],s[i] >= c[i]*tan(angmin[i]))
        @constraint(model,va_diff_max[i=1:n_buspairs],s[i] <= c[i]*tan(angmax[i]))
        if cut_voltage_angle_limits_model
            wf_lb = JuMP.lower_bound.(w_fr_bp)
            wf_ub = JuMP.upper_bound.(w_fr_bp)
            wt_lb = JuMP.lower_bound.(w_to_bp)
            wt_ub = JuMP.upper_bound.(w_to_bp)

            vf_lb, vf_ub = sqrt.(wf_lb), sqrt.(wf_ub)
            vt_lb, vt_ub = sqrt.(wt_lb), sqrt.(wt_ub)
            td_ub = angmax
            td_lb = angmin

            phi = (td_ub .+ td_lb)./2
            d   = (td_ub .- td_lb)./2

            sf = vf_lb .+ vf_ub
            st = vt_lb .+ vt_ub

            JuMP.@constraint(model,cut_va_1[i = 1:n_buspairs], sf[i]*st[i]*(cos(phi[i])*c[i] + sin(phi[i])*s[i]) - vt_ub[i]*cos(d[i])*st[i]*w_fr_bp[i] - vf_ub[i]*cos(d[i])*sf[i]*w_to_bp[i] >=  vf_ub[i]*vt_ub[i]*cos(d[i])*(vf_lb[i]*vt_lb[i] - vf_ub[i]*vt_ub[i]))
            JuMP.@constraint(model,cut_va_2[i = 1:n_buspairs], sf[i]*st[i]*(cos(phi[i])*c[i] + sin(phi[i])*s[i]) - vt_lb[i]*cos(d[i])*st[i]*w_fr_bp[i] - vf_lb[i]*cos(d[i])*sf[i]*w_to_bp[i] >= -vf_lb[i]*vt_lb[i]*cos(d[i])*(vf_lb[i]*vt_lb[i] - vf_ub[i]*vt_ub[i]))
        end
    end

    #################################################################################
    #NEURAL NET
    #################################################################################
    pg_input_inds = setdiff(pg_nonzero_inds,slack_gen_index)

    #Neural net inputs are model variables
    x = [pg[pg_input_inds];pd;qd[qd_nonzero_inds];w[bus_in_indices]]
    model.obj_dict[:x] = x

    #create output variables
    y = @variable(model,y[1:n_outputs])

    #Map neural net predictions to meaningful ACPF quantitites
    pref_out = y[pref_out_index]
    qg_out = y[qg_out_indices]
    w_out = y[w_out_indices]
    c_out = y[c_out_indices]
    s_out = y[s_out_indices]

    #Set up quantities to calculate violations
    qg_net = qg_out  #NN prediction
    pg_net = copy(pg)      #Model input
    pg_net[slack_gen_index] = pref_out #NN prediction

    w_net = Vector(undef,n_buses)
    w_net[bus_in_indices] .= w[bus_in_indices] #Model input
    w_net[bus_out_indices] .= w_out            #NN prediction
    w_net_fr_bp = [w_net[bus_pairs[i][1]] for i = 1:n_buspairs]
    w_net_to_bp = [w_net[bus_pairs[i][2]] for i = 1:n_buspairs]
    w_net_fr = w_net[bus_fr]
    w_net_to = w_net[bus_to]

    c_net = c_out #NN prediction
    s_net = s_out #NN prediction

    #Map c and s predictions to branches
    c_net_branch = Vector(undef,n_branches)
    s_net_branch = Vector(undef,n_branches)
    for i = 1:n_branches
        buspair = branch_to_buspair[i]
        c_net_branch[i] = c_net[pair_map[buspair]]
        s_net_branch[i] = s_net[pair_map[buspair]]
    end

    #Store quantities on the model
    model.obj_dict[:pref_out] = pref_out
    model.obj_dict[:qg_out] = qg_out
    model.obj_dict[:w_out] = w_out
    model.obj_dict[:c_out] = c_out
    model.obj_dict[:s_out] = s_out
    model.obj_dict[:pg_net] = pg_net

    #Add Neural Net Here: The magic happens
    NeuralOpt.add_neural_constraints!(model,x,y,chain,nn_mode;sparsify = true)

    #AC NEURAL NET FLOWS ARE EXPRESSIONS
    #neural net branch flows
    @expression(model,p_fr_net[i=1:n_branches],  (g[i]+g_fr[i])/tm[i]*w_net_fr[i] + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(c_net_branch[i]) + (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(s_net_branch[i]))
    @expression(model,q_fr_net[i=1:n_branches], -(b[i]+b_fr[i])/tm[i]*w_net_fr[i] - (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(c_net_branch[i]) + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(s_net_branch[i]))
    @expression(model,p_to_net[i=1:n_branches],  (g[i]+g_to[i])*w_net_to[i] + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(c_net_branch[i]) + (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(-s_net_branch[i]))
    @expression(model,q_to_net[i=1:n_branches], -(b[i]+b_to[i])*w_net_to[i] - (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(c_net_branch[i]) + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(-s_net_branch[i]))

    #neural net bus flows
    p_net = [p_fr_net;p_to_net]
    q_net = [q_fr_net;q_to_net]

    #These expressions consist entirely of model inputs and neural net outputs
    @expression(model,p_res[i=1:n_buses],sum(p_net[a] for a in bus_arcs[i]) - (sum(pg_net[g] for g in bus_gens[i]) - sum(pd[l] for l in bus_loads[i]) - sum(gs[s] for s in bus_shunts[i])*w_net[i]))
    @expression(model,q_res[i=1:n_buses],sum(q_net[a] for a in bus_arcs[i]) - (sum(qg_net[g] for g in bus_gens[i]) - sum(qd[l] for l in bus_loads[i]) + sum(bs[s] for s in bus_shunts[i])*w_net[i]))

    ###############################################
    #NEURAL NET VERIFIER CONSTRAINTS
    ###############################################
    if bound_slack_gen_nn
        JuMP.set_lower_bound(pref_out,dref[:gen][slack_gen_index]["pmin"])
        JuMP.set_upper_bound(pref_out,dref[:gen][slack_gen_index]["pmax"])
    end

    if bound_qg_nn
        for i = 1:n_gens
            JuMP.set_lower_bound(qg_net[i],dref[:gen][i]["qmin"])
            JuMP.set_upper_bound(qg_net[i],dref[:gen][i]["qmax"])
        end
    end

    if bound_output_bus_voltage_nn
        for i in bus_out_indices
            JuMP.set_lower_bound(w_net[i],dref[:bus][i]["vmin"]^2)
            JuMP.set_upper_bound(w_net[i],dref[:bus][i]["vmax"]^2)
        end
    end

    if bound_c_s_nn
        for i = 1:n_buspairs
            JuMP.set_lower_bound(c_net[i],c_lower[i])
            JuMP.set_upper_bound(c_net[i],c_upper[i])
            JuMP.set_lower_bound(s_net[i],s_lower[i])
            JuMP.set_upper_bound(s_net[i],s_upper[i])
        end
    end

    if thermal_limits_nn
        @constraint(model,thermal_fr_nn[i=1:n_branches],p_fr_net[i]^2 + q_fr_net[i]^2 <= dref[:branch][i]["rate_a"]^2)
        @constraint(model,thermal_to_nn[i=1:n_branches],p_to_net[i]^2 + q_to_net[i]^2 <= dref[:branch][i]["rate_a"]^2)
    end

    if voltage_angle_limits_nn
        angmin = [dref[:buspairs][bus_pairs[i]]["angmin"] for i = 1:n_buspairs]
        angmax = [dref[:buspairs][bus_pairs[i]]["angmax"] for i = 1:n_buspairs]
        @constraint(model,va_diff_min_nn[i=1:n_buspairs],s_net[i] >= c_net[i]*tan(angmin[i]))
        @constraint(model,va_diff_max_nn[i=1:n_buspairs],s_net[i] <= c_net[i]*tan(angmax[i]))
        if cut_voltage_angle_limits_nn
            wf_lb = JuMP.lower_bound.(w_net_fr_bp)
            wf_ub = JuMP.upper_bound.(w_net_fr_bp)
            wt_lb = JuMP.lower_bound.(w_net_to_bp)
            wt_ub = JuMP.upper_bound.(w_net_to_bp)

            vf_lb, vf_ub = sqrt.(wf_lb), sqrt.(wf_ub)
            vt_lb, vt_ub = sqrt.(wt_lb), sqrt.(wt_ub)
            td_ub = angmax
            td_lb = angmin

            phi = (td_ub .+ td_lb)./2
            d   = (td_ub .- td_lb)./2

            sf = vf_lb .+ vf_ub
            st = vt_lb .+ vt_ub

            JuMP.@constraint(model,cut_va_1_nn[i = 1:n_buspairs], sf[i]*st[i]*(cos(phi[i])*c[i] + sin(phi[i])*s[i]) - vt_ub[i]*cos(d[i])*st[i]*w_net_fr_bp[i] - vf_ub[i]*cos(d[i])*sf[i]*w_net_to_bp[i] >=  vf_ub[i]*vt_ub[i]*cos(d[i])*(vf_lb[i]*vt_lb[i] - vf_ub[i]*vt_ub[i]))
            JuMP.@constraint(model,cut_va_2_nn[i = 1:n_buspairs], sf[i]*st[i]*(cos(phi[i])*c[i] + sin(phi[i])*s[i]) - vt_lb[i]*cos(d[i])*st[i]*w_net_fr_bp[i] - vf_lb[i]*cos(d[i])*sf[i]*w_net_to_bp[i] >= -vf_lb[i]*vt_lb[i]*cos(d[i])*(vf_lb[i]*vt_lb[i] - vf_ub[i]*vt_ub[i]))
        end
    end

    #Gen power factor
    if gen_power_factor_nn == :constant  #Assume perfect generators
        @constraint(model,pfgen_nn[i in pg_nonzero_inds],pg_net[i]^2 == (pg_net[i]^2 + qg_net[i]^2))
    elseif gen_power_factor_nn == :minimum
        @constraint(model,pfgen_nn[i in pg_nonzero_inds],pg_net[i]^2 >= 0.95^2*(pg_net[i]^2 + qg_net[i]^2))
    end

    #NN must satisfy physics relaxation
    if soc_nn
        @constraint(model,soc_nn[i=1:n_buspairs], c_net[i]^2 + s_net[i]^2 <= w_net_fr_bp[i]*w_net_to_bp[i])
    end

    return model
end

#Local verifier that uses complementarity constraints for ReLU
#This verifier represents the true ACPF physics using w,c,s and theta. It is a nonlinear model, so really only useful for local verification
function build_acpf_wcs_theta_verifier(dref::Dict,chain;nn_mode = NeuralOpt.ComplementarityReluMode(1e-8),
    pg_delta = 0.1,pd_delta = 0.1,w_delta = 0.1,
    constant_load_power_factor = true,
    thermal_limits_model = false,
    voltage_angle_limits_model = false,
    bound_slack_gen_model = false,
    bound_output_bus_voltage_model = false,
    bound_c_s_model = false,
    bound_qg_model = false)

    n_buses = length(dref[:bus])
    n_gens = length(dref[:gen])
    n_loads = length(dref[:load])
    n_shunts = length(dref[:shunt])
    n_buspairs = length(dref[:buspairs])
    n_branches = length(dref[:branch])

    #SHUNTS
    bs,gs = load_shunt_data(dref)

    #BRANCH DATA
    g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to = load_branch_data(dref)
    c_lower,c_upper,s_lower,s_upper = load_w_bounds(dref)
    bus_loads,load_to_bus,bus_gens,gen_to_bus,bus_shunts,shunt_to_bus,bus_arcs,arc_to_bus = load_mappings(dref)

    #WCS INPUT AND OUTPUT INDICES for ACPF EQUATIONS
    pg_in_indices,qd_in_indices,pd_in_indices,w_in_indices =  map_wcs_inputs(dref)
    pref_out_index,qg_out_indices,w_out_indices,c_out_indices,s_out_indices = map_wcs_outputs(dref) # w_set are voltage setpoints.  w_out for each bus are the outputs
    va_slack,slack_index = load_slack(dref)
    ########################################
    #Check input and output dimensions
    ########################################
    input_layer = chain.layers[1]
    output_layer = chain.layers[end]
    n_inputs = size(input_layer.W)[2]
    n_outputs = size(output_layer.W)[1]

    @assert n_inputs == length(pd_in_indices) + length(qd_in_indices) + length(pg_in_indices) + length(w_in_indices)
    @assert n_outputs == length(w_out_indices) + length(c_out_indices) + length(s_out_indices) + length(qg_out_indices) + 1 #The 1 is pref generator

    #########################################
    # Model Topology-Based Indices
    #########################################
    slack_gen_index = get_slack_gen_index(dref)
    qd_zero_inds,pg_zero_inds,qd_nonzero_inds,pg_nonzero_inds = get_fixed_gen_load_indices(dref)
    bus_in_indices,bus_out_indices = get_in_out_bus_indices(dref)
    bus_pairs,branch_to_buspair = get_buspairs(dref)
    bus_pairs = sort(bus_pairs)

    #LOAD POWER FACTORS IN MATPOWER
    pd_data = [dref[:load][i]["pd"] for i = 1:n_loads]
    qd_data = [dref[:load][i]["qd"] for i = 1:n_loads]
    S_data = sqrt.(pd_data.^2 + qd_data.^2)
    pf = pd_data ./ S_data
    pf_coeff = sqrt.((1.0 .- pf.^2)./pf.^2)

    #########################################
    # Create Model
    #########################################
    model = Model()

    #VOLTAGES
    @variable(model, w[i in 1:n_buses],start = 1.001)
    for i in bus_in_indices
        JuMP.set_lower_bound(w[i],dref[:bus][i]["vm"]^2*(1-w_delta))
        JuMP.set_upper_bound(w[i],dref[:bus][i]["vm"]^2*(1+w_delta))
    end
    for i in bus_out_indices
        JuMP.set_lower_bound(w[i],0)
    end
    @variable(model, theta[i in 1:n_buses],start = 0.0)# start = dref[:bus][i]["va"])
    @constraint(model,slack_angle,theta[slack_index] == va_slack)
    @variable(model, c[i in 1:n_buspairs], start = 1.0)
    @variable(model, s[i in 1:n_buspairs], start = 0.01)

    #POWER GENERATION
    @variable(model, pg[i in 1:n_gens])
    for i in pg_zero_inds
        JuMP.fix(pg[i],0;force = true)
    end
    for i in pg_nonzero_inds
        if i != slack_gen_index
            JuMP.set_lower_bound(pg[i],dref[:gen][i]["pg"]*(1-pg_delta))
            JuMP.set_upper_bound(pg[i],dref[:gen][i]["pg"]*(1+pg_delta))
        end
    end
    @variable(model, qg[i in 1:n_gens])

    #LOADS
    @variable(model, pd[i in 1:n_loads])
    for i = 1:n_loads
        JuMP.set_lower_bound(pd[i],dref[:load][i]["pd"]*(1-pd_delta))
        JuMP.set_upper_bound(pd[i],dref[:load][i]["pd"]*(1+pd_delta))
    end
    @variable(model, qd[i in 1:n_loads])
    if constant_load_power_factor
        @constraint(model,pfcon[i in qd_nonzero_inds],model[:qd][i] == pf_coeff[i]*model[:pd][i])  #NOTE: assumes positive reactive power
    end
    for i in qd_zero_inds
        JuMP.fix(qd[i],0;force = true)
    end

    #OPERATIONAL CONSTRAINTS
    if bound_slack_gen_model
        JuMP.set_lower_bound(pg[slack_gen_index],dref[:gen][slack_gen_index]["pmin"])
        JuMP.set_upper_bound(pg[slack_gen_index],dref[:gen][slack_gen_index]["pmax"])
    end

    if bound_qg_model
        for i = 1:n_gens
            JuMP.set_lower_bound(qg[i],dref[:gen][i]["qmin"])
            JuMP.set_upper_bound(qg[i],dref[:gen][i]["qmax"])
        end
    end

    if bound_output_bus_voltage_model
        for i in bus_out_indices
            JuMP.set_lower_bound(w[i],dref[:bus][i]["vmin"]^2)
            JuMP.set_upper_bound(w[i],dref[:bus][i]["vmax"]^2)
        end
    end

    ###############################################
    #AC TRUE BRANCH FLOWS ARE CONSTRAINTS
    ###############################################
    w_fr_bp = [w[bus_pairs[i][1]] for i = 1:n_buspairs]
    w_to_bp = [w[bus_pairs[i][2]] for i = 1:n_buspairs]
    theta_fr_bp = [theta[bus_pairs[i][1]] for i = 1:n_buspairs]
    theta_to_bp = [theta[bus_pairs[i][2]] for i = 1:n_buspairs]

    #w for branch flows
    w_fr = w[bus_fr]
    w_to = w[bus_to]

    #MAP BETWEEN BRANCHES AND BUSPAIRS
    pair_map = Dict()
    for (i,pair) in enumerate(bus_pairs)
        pair_map[pair] = i
    end
    s_branch = Vector(undef,n_branches)
    c_branch = Vector(undef,n_branches)
    for i = 1:n_branches
        buspair = branch_to_buspair[i]
        c_branch[i] = c[pair_map[buspair]]
        s_branch[i] = s[pair_map[buspair]]
    end

    ############################################################################################################################################################################################################################################
    @expression(model,p_fr[i=1:n_branches],(g[i]+g_fr[i])/tm[i]*w_fr[i] + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(c_branch[i]) + (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(s_branch[i]))
    @expression(model,q_fr[i=1:n_branches],-(b[i]+b_fr[i])/tm[i]*w_fr[i] - (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(c_branch[i]) + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(s_branch[i]))
    @expression(model,p_to[i=1:n_branches],(g[i]+g_to[i])*w_to[i] + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(c_branch[i]) + (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(-s_branch[i]))
    @expression(model,q_to[i=1:n_branches],-(b[i]+b_to[i])*w_to[i] - (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(c_branch[i]) + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(-s_branch[i]))

    #POWER BALANCE
    p = [p_fr;p_to]
    q = [q_fr;q_to]
    @constraint(model,active_power[i=1:n_buses],sum(p[a] for a in bus_arcs[i])  == sum(pg[g] for g in bus_gens[i]) - sum(pd[l] for l in bus_loads[i]) - sum(gs[s] for s in bus_shunts[i])*w[i])
    @constraint(model,reactive_power[i=1:n_buses],sum(q[a] for a in bus_arcs[i])  == sum(qg[g] for g in bus_gens[i]) - sum(qd[l] for l in bus_loads[i]) + sum(bs[s] for s in bus_shunts[i])*w[i])

    #"SOC" constraint
    @constraint(model,soc[i=1:n_buspairs], c[i]^2 + s[i]^2 == w_fr_bp[i]*w_to_bp[i])

    #Cycle constraint
    @NLconstraint(model,cycle[i=1:n_buspairs],s[i] == c[i]*tan(theta_fr_bp[i] - theta_to_bp[i]))

    if thermal_limits_model
        @constraint(model,thermal_fr[i=1:n_branches],p_fr[i]^2 + q_fr[i]^2 <= dref[:branch][i]["rate_a"]^2)
        @constraint(model,thermal_to[i=1:n_branches],p_to[i]^2 + q_to[i]^2 <= dref[:branch][i]["rate_a"]^2)
    end

    if voltage_angle_limits_model
        angmin = [dref[:buspairs][bus_pairs[i]]["angmin"] for i = 1:n_buspairs]
        angmax = [dref[:buspairs][bus_pairs[i]]["angmax"] for i = 1:n_buspairs]
        @constraint(model,va_diff_min[i=1:n_buspairs],(theta_fr_bp[i] - theta_to_bp[i]) >= angmin[i])
        @constraint(model,va_diff_max[i=1:n_buspairs],(theta_fr_bp[i] - theta_to_bp[i]) <= angmax[i])
    end
    ############################################################################################################################################################################################################################################

    ###############################################
    #AC NEURAL NET BRANCH FLOWS
    pg_input_inds = setdiff(pg_nonzero_inds,slack_gen_index)
    x = [pg[pg_input_inds];pd;qd[qd_nonzero_inds];w[bus_in_indices]]
    model.obj_dict[:x] = x

    #map variables to outputs
    y = @variable(model,y[1:n_outputs])

    pref_out = y[pref_out_index]
    qg_out = y[qg_out_indices]
    w_out = y[w_out_indices]

    c_out = y[c_out_indices]
    s_out = y[s_out_indices]

    qg_net = qg_out
    pg_net =  pg
    pg_net[slack_gen_index] = pref_out

    w_net = Vector(undef,n_buses)
    w_net[bus_in_indices] .= w[bus_in_indices]
    w_net[bus_out_indices] .= w_out
    w_net_fr = w_net[bus_fr]
    w_net_to = w_net[bus_to]


    c_net = c_out
    s_net = s_out
    c_net_branch = Vector(undef,n_branches)
    s_net_branch = Vector(undef,n_branches)
    for i = 1:n_branches
        buspair = branch_to_buspair[i]
        c_net_branch[i] = c_net[pair_map[buspair]]
        s_net_branch[i] = s_net[pair_map[buspair]]
    end

    model.obj_dict[:pref_out] = pref_out
    model.obj_dict[:qg_out] = qg_out
    model.obj_dict[:w_out] = w_out
    model.obj_dict[:c_out] = c_out
    model.obj_dict[:s_out] = s_out
    model.obj_dict[:pg_net] = pg_net

    #Add Neural Net Here:
    add_neural_constraints!(model,x,y,chain,nn_mode;sparsify = true)

    #AC NEURAL NET FLOWS ARE EXPRESSIONS
    ###############################################
    #neural net branch flows
    @expression(model,p_fr_net[i=1:n_branches],  (g[i]+g_fr[i])/tm[i]*w_net_fr[i] + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(c_net_branch[i]) + (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(s_net_branch[i]))
    @expression(model,q_fr_net[i=1:n_branches], -(b[i]+b_fr[i])/tm[i]*w_net_fr[i] - (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(c_net_branch[i]) + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(s_net_branch[i]))
    @expression(model,p_to_net[i=1:n_branches],  (g[i]+g_to[i])*w_net_to[i] + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(c_net_branch[i]) + (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(-s_net_branch[i]))
    @expression(model,q_to_net[i=1:n_branches], -(b[i]+b_to[i])*w_net_to[i] - (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(c_net_branch[i]) + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(-s_net_branch[i]))

    #neural net bus flows
    p_net = [p_fr_net;p_to_net]
    q_net = [q_fr_net;q_to_net]
    @expression(model,p_res[i=1:n_buses],sum(p_net[a] for a in bus_arcs[i]) - (sum(pg_net[g] for g in bus_gens[i]) - sum(pd[l] for l in bus_loads[i]) - sum(gs[s] for s in bus_shunts[i])*w_net[i]))
    @expression(model,q_res[i=1:n_buses],sum(q_net[a] for a in bus_arcs[i]) - (sum(qg_net[g] for g in bus_gens[i]) - sum(qd[l] for l in bus_loads[i]) + sum(bs[s] for s in bus_shunts[i])*w_net[i]))
    ###############################################

    return model
end

#Given a vector of ACPF inputs, compute the true nonlinear ACPF solution and calculate output errors
#This model is used to check the ACPF solution obtained using the neural network. 
function build_acpf_input_verifier(dref::Dict,input::Vector)
    model = Model()
    n_buses = length(dref[:bus])
    n_gens = length(dref[:gen])
    n_loads = length(dref[:load])
    n_shunts = length(dref[:shunt])
    n_branches = length(dref[:branch])

    #SHUNTS
    bs,gs = load_shunt_data(dref)
    va_slack,slack_index = load_slack(dref)
    slack_gen_index = get_slack_gen_index(dref)
    qd_zero_inds,pg_zero_inds,qd_nonzero_inds,pg_nonzero_inds = get_fixed_gen_load_indices(dref)
    pg_input_inds = setdiff(pg_nonzero_inds,slack_gen_index)
    #BRANCH DATA
    g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to = load_branch_data(dref)

    #MAP neural net outputs back to bus indices.
    pg_in_indices,pd_in_indices,qd_in_indices,w_in_indices =  map_wcs_inputs(dref)
    #vm_map,va_map,vm_out_indices,va_out_indices,qg_out_indices,pref_out_index = map_net_outputs(dref)

    bus_loads,load_to_bus,bus_gens,gen_to_bus,bus_shunts,shunt_to_bus,bus_arcs,arc_to_bus = load_mappings(dref)
    bus_in_indices,bus_out_indices = get_in_out_bus_indices(dref)

    #VOLTAGE
    @variable(model, vm[1:n_buses], start = 1.0)
    @variable(model, va[1:n_buses], start = 1.0)
    @constraint(model,slack_angle,va[slack_index] == va_slack)

    #POWER GENERATION
    @variable(model, pg[i in 1:n_gens] )
    @variable(model, qg[i in 1:n_gens] )

    #LOADS (NOTE: We seek to find loads that maximize violation)
    @variable(model, pd[1:n_loads])
    @variable(model, qd[1:n_loads])

    #Fix inputs
    for i in qd_zero_inds
        JuMP.fix(qd[i],0)
    end
    for i in pg_zero_inds
        JuMP.fix(pg[i],0)
    end

    pg_input = input[pg_in_indices]
    pd_input = input[pd_in_indices]
    qd_input = input[qd_in_indices]
    wi_input = input[w_in_indices]

    for i = 1:length(pg_in_indices)
        pg_ind = pg_input_inds[i]
        JuMP.fix(pg[pg_ind],pg_input[i])
    end

    for i = 1:length(pd_in_indices)
        JuMP.fix(pd[i],pd_input[i])
    end

    for i = 1:length(qd_in_indices)
        qd_ind = qd_nonzero_inds[i]
        JuMP.fix(qd[qd_ind],qd_input[i])
    end

    for i = 1:length(w_in_indices)
        v_ind = bus_in_indices[i]
        JuMP.fix(vm[v_ind],sqrt(wi_input[i]))
    end

    ###############################################
    vm_fr = vm[bus_fr]
    vm_to = vm[bus_to]
    va_fr = va[bus_fr]
    va_to = va[bus_to]

    #per unit branch flow expressions
    @NLexpression(model,p_fr[i = 1:n_branches],(g[i]+g_fr[i])/tm[i]*vm_fr[i]^2 + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(vm_fr[i]*vm_to[i]*cos(va_fr[i]-va_to[i])) + (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(vm_fr[i]*vm_to[i]*sin(va_fr[i]-va_to[i])))
    @NLexpression(model,q_fr[i = 1:n_branches],-(b[i]+b_fr[i])/tm[i]*vm_fr[i]^2 - (-b[i]*tr[i]-g[i]*ti[i])/tm[i]*(vm_fr[i]*vm_to[i]*cos(va_fr[i]-va_to[i])) + (-g[i]*tr[i]+b[i]*ti[i])/tm[i]*(vm_fr[i]*vm_to[i]*sin(va_fr[i]-va_to[i])))
    @NLexpression(model,p_to[i = 1:n_branches],(g[i]+g_to[i])*vm_to[i]^2 + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(vm_to[i]*vm_fr[i]*cos(va_to[i]-va_fr[i])) + (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(vm_to[i]*vm_fr[i]*sin(va_to[i]-va_fr[i])) )
    @NLexpression(model,q_to[i = 1:n_branches],-(b[i]+b_to[i])*vm_to[i]^2 - (-b[i]*tr[i]+g[i]*ti[i])/tm[i]*(vm_to[i]*vm_fr[i]*cos(va_fr[i]-va_to[i])) + (-g[i]*tr[i]-b[i]*ti[i])/tm[i]*(vm_to[i]*vm_fr[i]*sin(va_to[i]-va_fr[i])) )

    #POWER BALANCE
    p = [p_fr;p_to]
    q = [q_fr;q_to]
    @NLconstraint(model,active_power[i=1:n_buses],sum(p[a] for a in bus_arcs[i])  == sum(pg[g] for g in bus_gens[i]) - sum(pd[l] for l in bus_loads[i]) - sum(gs[s] for s in bus_shunts[i])*vm[i]^2)
    @NLconstraint(model,reactive_power[i=1:n_buses],sum(q[a] for a in bus_arcs[i])  == sum(qg[g] for g in bus_gens[i]) - sum(qd[l] for l in bus_loads[i]) + sum(bs[s] for s in bus_shunts[i])*vm[i]^2)

    return model
end
