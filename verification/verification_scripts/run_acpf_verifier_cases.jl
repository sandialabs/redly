#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#this script runs verification problems with different constraints active.
using Gurobi
const GRB_ENV = Gurobi.Env()

using Formatting
using JSON, CSV
using DataFrames, Statistics
using JuMP
#load models
include((@__DIR__)*"/acpf_verifiers.jl")
include((@__DIR__)*"/../json_utils.jl")

#Load case
n_bus = 14
# n_bus = 118
sparsity_run = 0
f = open("../../src/acpf/ieee_case$(n_bus).json", "r")
config = JSON.parse(f)
close(f)

name = config["name"]
b_p = sprintf1("%.2g", config["b_p"])
alpha = sprintf1("%.2g", config["alpha"])
dropout = sprintf1("%.2g", config["dropout"])
ld_int = sprintf1("%d", config["ld_int"])
ld_prog = sprintf1("%d", config["ld_prog"])
epochs = sprintf1("%d", config["epochs"])

folder = "../../output/ieee_case$(n_bus)"
prefixp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(ld_int)_ldp_$(ld_prog)_run_$(sparsity_run)"
prefixnp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(epochs)_ldp_$(ld_prog)_run_$(sparsity_run)"

case = "pglib_opf_case$(n_bus)_ieee.m"
data = PowerModels.parse_file("../../data/ieee_case$(n_bus)/$(case)")
dref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

n_buses = length(dref[:bus])
n_loads = length(dref[:load])
n_gens = length(dref[:gen])
n_buspairs = length(dref[:buspairs])

g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to = load_branch_data(dref)
bus_pairs,branch_to_buspair = get_buspairs(dref)
slack_gen_index = get_slack_gen_index(dref)
bus_in_indices,bus_out_indices = get_in_out_bus_indices(dref)
qd_zero_inds,pg_zero_inds,qd_nonzero_inds,pg_nonzero_inds = get_fixed_gen_load_indices(dref)

#Set nominal operating point
df = CSV.read("../../data/ieee_case$(n_bus)/input_0.csv", DataFrame)
cols = names(df)

pd_delta = 0.25   #0.25
w_delta = 0.05    #0.05
pg_delta = 0.25   #0.25

qd_min = []
qd_max = []
pd_nominal = []
qd_nominal = []
pg_nominal = []
vm_nominal = []
qd_dict = Dict()
for load_idx = 1:length(dref[:load])
    load = dref[:load][load_idx]
    bus_idx = load["load_bus"]
    data_pd = df[!, Symbol("pl:b$(bus_idx)")]
    dref[:load][load_idx]["pd"] = mean(data_pd)
    data_qd = df[!, Symbol("ql:b$(bus_idx)")]
    qd_dict[load_idx] = data_qd

    #NOTE: be careful changing this. It determines the nominal load power factor.
    #dref[:load][load_idx]["qd"] = mean(data_qd)
    push!(qd_min,minimum(data_qd))
    push!(qd_max,maximum(data_qd))
    push!(pd_nominal,mean(data_pd))
    push!(qd_nominal,mean(data_qd))
end
for gen_idx = 1:length(dref[:gen])
    gen = dref[:gen][gen_idx]
    # for (gen_idx,gen) in dref[:gen]
    bus_idx = gen["gen_bus"]
    if dref[:bus][bus_idx]["bus_type"] != 3
      data = df[!, Symbol("pg:b$(bus_idx),g$(gen_idx)")]
      dref[:gen][gen_idx]["pg"] = mean(data)
      push!(pg_nominal,mean(data))
    end
end
for bus_idx = 1:length(dref[:bus])
     bt = dref[:bus][bus_idx]["bus_type"]
     if bt == 2
         data = df[!, Symbol("vm:b$(bus_idx)")]
     elseif bt == 3
         data = df[!, Symbol("vm:$(bus_idx),ref")]
     else
         continue
     end
     dref[:bus][bus_idx]["vm"] = mean(data)
     push!(vm_nominal,mean(data))
end

case1 = [true,:off,:off,true,true,true,false,false,false,false,:off,false,false,false,false,false,false,false,false]
case3 = [false,:constant,:off,true,true,true,false,false,false,false,:off,false,false,false,false,false,false,false,false]
case4 = [false,:constant,:minimum,true,true,true,true,true,true,true,:off,false,false,false,false,false,false,false,false]
case6 = [false,:constant,:minimum,true,true,true,true,true,true,true,:minimum,true,true,true,true,true,true,true,false]
cases = [case1,case3,case4,case6]

##############################################################
#Run sweep over constraints with mae formulation
##############################################################
result_dir = folder*"/constant_load_pf/violations_gurobi_pd_$(Int(pd_delta*100))_w_$(Int(w_delta*100))_pg_$(Int(pg_delta*100))"
mkpath(result_dir)

##############################################################
#Run verification on a single bus
##############################################################
prefixes = [prefixp, prefixnp]
for prefix in prefixes
    net_file = folder*"/$(prefix)_model_data.json"
    chain = read_acpf_json(net_file)    
    
    p_res_status_codes = []
    q_res_status_codes = []
    p_res_bounds = []
    q_res_bounds = []
    p_res_mae = []
    q_res_mae = []
    p_res_solve_times = []
    q_res_solve_times = []
    for (i,case) in enumerate(cases)
        model = build_acpf_wcs_verifier(dref,chain;nn_mode = NeuralOpt.IndicatorReluMode(),
        pg_delta = pg_delta,pd_delta = pd_delta,w_delta = w_delta,qd_delta = :off,
        qd_min = qd_min,qd_max = qd_max,
        qd_limits = case[1],
        load_power_factor = case[2],
        gen_power_factor_model = case[3],
        bound_slack_gen_model = case[4],
        bound_qg_model = case[5],
        bound_output_bus_voltage_model = case[6],
        bound_c_s_model = case[7],
        thermal_limits_model = case[8],
        voltage_angle_limits_model = case[9],
        cut_voltage_angle_limits_model = case[10],
        gen_power_factor_nn = case[11],
        bound_slack_gen_nn = case[12],
        bound_qg_nn = case[13],
        bound_output_bus_voltage_nn = case[14],
        bound_c_s_nn = case[15],
        thermal_limits_nn = case[16],
        voltage_angle_limits_nn = case[17],
        cut_voltage_angle_limits_nn = case[18],
        soc_nn = case[19])
    
        #gurobi = Gurobi.Optimizer
        #set_optimizer(model,gurobi)
        set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))
        set_optimizer_attribute(model,"NonConvex",2)
        set_optimizer_attribute(model,"PSDTol",1e-6)
        set_optimizer_attribute(model,"Threads",16)
        #set_optimizer_attribute(model,"Threads",1)
        set_optimizer_attribute(model,"TimeLimit",1000)
    
        # Indicator approach
        @variable(model,y_abs[i=1:n_buses])
        @variable(model,z1[1:n_buses],Bin)
        @constraint(model,indicator_max_1[i = 1:n_buses],!z1[i] => {y_abs[i] == model[:p_res][i]})  #z1 = 0 -> max p_res is positive
        @constraint(model,indicator_max_2[i = 1:n_buses], z1[i] => {y_abs[i] == -model[:p_res][i]})
        @objective(model,Max,sum(y_abs)/n_buses)
        optimize!(model)
    
        push!(p_res_bounds,objective_bound(model))
        push!(p_res_solve_times,JuMP.solve_time(model))
        push!(p_res_status_codes,termination_status(model))
        if has_values(model)
           push!(p_res_mae,objective_value(model))
        else
           push!(p_res_mae,Inf)
        end
    end
    
    for (i,case) in enumerate(cases)
        println(i)
        model = build_acpf_wcs_verifier(dref,chain;nn_mode = NeuralOpt.IndicatorReluMode(),
        pg_delta = pg_delta,pd_delta = pd_delta,w_delta = w_delta,qd_delta = :off,
        qd_min = qd_min,qd_max = qd_max,
        qd_limits = case[1],
        load_power_factor = case[2],
        gen_power_factor_model = case[3],
        bound_slack_gen_model = case[4],
        bound_qg_model = case[5],
        bound_output_bus_voltage_model = case[6],
        bound_c_s_model = case[7],
        thermal_limits_model = case[8],
        voltage_angle_limits_model = case[9],
        cut_voltage_angle_limits_model = case[10],
        gen_power_factor_nn = case[11],
        bound_slack_gen_nn = case[12],
        bound_qg_nn = case[13],
        bound_output_bus_voltage_nn = case[14],
        bound_c_s_nn = case[15],
        thermal_limits_nn = case[16],
        voltage_angle_limits_nn = case[17],
        cut_voltage_angle_limits_nn = case[18],
        soc_nn = case[19])
    
        #MAE objective
        #gurobi = Gurobi.Optimizer
        #set_optimizer(model,gurobi)
        set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))
        set_optimizer_attribute(model,"NonConvex",2)
        set_optimizer_attribute(model,"PSDTol",1e-6)
        set_optimizer_attribute(model,"Threads",16)
        #set_optimizer_attribute(model,"Threads",1)
        set_optimizer_attribute(model,"TimeLimit",1000)
    
        # Indicator approach
        @variable(model,y_abs[i=1:n_buses])#,start = value(model_ipopt[:p_res][i]))
        @variable(model,z1[1:n_buses],Bin)
        @constraint(model,indicator_max_1[i = 1:n_buses],!z1[i] => {y_abs[i] == model[:q_res][i]})  #z1 = 0 -> max p_res is positive
        @constraint(model,indicator_max_2[i = 1:n_buses], z1[i] => {y_abs[i] == -model[:q_res][i]})
        @objective(model,Max,sum(y_abs)/n_buses)
        optimize!(model)
    
        push!(q_res_bounds,objective_bound(model))
        push!(q_res_solve_times,JuMP.solve_time(model))
        push!(q_res_status_codes,termination_status(model))
        if has_values(model)
           push!(q_res_mae,objective_value(model))
        else
           push!(q_res_mae,Inf)
        end
    end
    
    results = Dict("p_res_mae" => p_res_mae,"q_res_mae" => q_res_mae,"cases" => cases,"p_res_status_codes" => p_res_status_codes,
    "q_res_status_codes" => q_res_status_codes,"p_res_solve_times"=>p_res_solve_times,"q_res_solve_times"=> q_res_solve_times)
    json_string = JSON.json(results)
    open(result_dir*"/$(prefix)_bus_errors_mae_allcases.json","w") do f
        write(f, json_string)
    end
end

##############################################################
#Run sweep over constraints with max formulation
##############################################################
for prefix in prefixes
    case_index = 0
    for case in cases
        case_index += 1
        net_file = folder*"/$(prefix)_model_data.json"
        chain = read_acpf_json(net_file)

        p_res_max = []
        p_res_bound = []
        q_res_max = []
        q_res_bound = []
        x_input_p_res = []
        y_output_p_res = []
        x_input_q_res = []
        y_output_q_res = []
        status_p_res = []
        status_q_res = []
        p_solve_times = []
        q_solve_times = []
        for j = 1:n_buses
            model = build_acpf_wcs_verifier(dref,chain;nn_mode = NeuralOpt.IndicatorReluMode(),
            pg_delta = pg_delta,pd_delta = pd_delta,w_delta = w_delta,qd_delta = :off,
            qd_min = qd_min,qd_max = qd_max,
            qd_limits = case[1],
            load_power_factor = case[2],
            gen_power_factor_model = case[3],
            bound_slack_gen_model = case[4],
            bound_qg_model = case[5],
            bound_output_bus_voltage_model = case[6],
            bound_c_s_model = case[7],
            thermal_limits_model = case[8],
            voltage_angle_limits_model = case[9],
            cut_voltage_angle_limits_model = case[10],
            gen_power_factor_nn = case[11],
            bound_slack_gen_nn = case[12],
            bound_qg_nn = case[13],
            bound_output_bus_voltage_nn = case[14],
            bound_c_s_nn = case[15],
            thermal_limits_nn = case[16],
            voltage_angle_limits_nn = case[17],
            cut_voltage_angle_limits_nn = case[18],
            soc_nn = case[19])

            #gurobi = Gurobi.Optimizer
            #set_optimizer(model,gurobi)
            set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))
            set_optimizer_attribute(model,"NonConvex",2)
            set_optimizer_attribute(model,"PSDTol",1e-6)
            set_optimizer_attribute(model,"Threads",16)
            #set_optimizer_attribute(model,"Threads",1)
            set_optimizer_attribute(model,"TimeLimit",7200)

            @variable(model,y_abs)
            @variable(model,z,Bin)
            @constraint(model,indicator_max_1,!z => {y_abs == model[:p_res][j]})  #z1 = 0 -> max p_res is positive
            @constraint(model,indicator_max_2, z => {y_abs == -model[:p_res][j]})
            @objective(model,Max,y_abs)
            optimize!(model)
            status = termination_status(model)
            push!(status_p_res,status)

            p_res = value(model[:p_res][j])
            x = value.(model[:x])
            y = value.(model[:y])
            p_bound = objective_bound(model)
            solution_time = JuMP.solve_time(model)

            push!(p_res_max,p_res)
            push!(p_res_bound,p_bound)
            push!(x_input_p_res,x)
            push!(y_output_p_res,y)
            push!(p_solve_times,solution_time)
        end
        for j = 1:n_buses
            model = build_acpf_wcs_verifier(dref,chain;nn_mode = NeuralOpt.IndicatorReluMode(),
            pg_delta = pg_delta,pd_delta = pd_delta,w_delta = w_delta,qd_delta = :off,
            qd_min = qd_min,qd_max = qd_max,
            qd_limits = case[1],
            load_power_factor = case[2],
            gen_power_factor_model = case[3],
            bound_slack_gen_model = case[4],
            bound_qg_model = case[5],
            bound_output_bus_voltage_model = case[6],
            bound_c_s_model = case[7],
            thermal_limits_model = case[8],
            voltage_angle_limits_model = case[9],
            cut_voltage_angle_limits_model = case[10],
            gen_power_factor_nn = case[11],
            bound_slack_gen_nn = case[12],
            bound_qg_nn = case[13],
            bound_output_bus_voltage_nn = case[14],
            bound_c_s_nn = case[15],
            thermal_limits_nn = case[16],
            voltage_angle_limits_nn = case[17],
            cut_voltage_angle_limits_nn = case[18],
            soc_nn = case[19])

            #gurobi = Gurobi.Optimizer
            #set_optimizer(model,gurobi)
            set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))
            set_optimizer_attribute(model,"NonConvex",2)
            set_optimizer_attribute(model,"PSDTol",1e-6)
            set_optimizer_attribute(model,"Threads",16)
            #set_optimizer_attribute(model,"Threads",1)
            set_optimizer_attribute(model,"TimeLimit",7200)

            @variable(model,y_abs)
            @variable(model,z,Bin)
            @constraint(model,indicator_max_1,!z => {y_abs == model[:q_res][j]})
            @constraint(model,indicator_max_2, z => {y_abs == -model[:q_res][j]})
            @objective(model,Max,y_abs)
            optimize!(model)

            status = termination_status(model)
            push!(status_q_res,status)
            q_res = value(model[:q_res][j])
            x = value.(model[:x])
            y = value.(model[:y])
            q_bound = objective_bound(model)
            solution_time = JuMP.solve_time(model)

            push!(q_res_bound,q_bound)
            push!(q_res_max,q_res)
            push!(x_input_q_res,x)
            push!(y_output_q_res,y)
            push!(q_solve_times,solution_time)
        end
        max_results = Dict("p_res_max" => p_res_max,"q_res_max" => q_res_max,"x_input_p_res" => x_input_p_res,"y_output_p_res" => y_output_p_res,
        "x_input_q_res" => x_input_q_res,"y_output_q_res" => y_output_q_res,"p_res_status" => status_p_res,"q_res_status" => status_q_res,
        "p_solve_times" => p_solve_times,"q_solve_times"=> q_solve_times)
        json_string = JSON.json(max_results)

        open(result_dir*"/$(prefix)_bus_errors_mae_case_$(case_index).json","w") do f
            write(f, json_string)
        end
    end
end
