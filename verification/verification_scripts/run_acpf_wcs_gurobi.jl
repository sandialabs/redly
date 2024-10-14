#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#run verification for each bus in the system on active and reactive power
#run MAE verification
using Gurobi
const GRB_ENV = Gurobi.Env()

using Formatting
using JSON
using CSV
using DataFrames
using Statistics

include((@__DIR__)*"/acpf_verifiers.jl")
include((@__DIR__)*"/../json_utils.jl")

#Load case
n_bus = 14
# n_bus = 118
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
runs = sprintf1("%d", config["runs"])

folder = "../../output/ieee_case$(n_bus)"

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
for (load_idx,load) in dref[:load]
    bus_idx = load["load_bus"]
    data = df[!, Symbol("pl:b$(bus_idx)")]
    dref[:load][load_idx]["pd"] = mean(data)
end
for (gen_idx,gen) in dref[:gen]
    bus_idx = gen["gen_bus"]
    if dref[:bus][bus_idx]["bus_type"] != 3
        data = df[!, Symbol("pg:b$(bus_idx),g$(gen_idx)")]
        dref[:gen][gen_idx]["pg"] = mean(data)
    end
end
for (bus_idx,bus) in dref[:bus]
    bt = dref[:bus][bus_idx]["bus_type"]
    if bt == 2
        data = df[!, Symbol("vm:b$(bus_idx)")]
    elseif bt == 3
        data = df[!, Symbol("vm:$(bus_idx),ref")]
    else
        continue
    end
    dref[:bus][bus_idx]["vm"] = mean(data)
end

pd_delta = 0.25   #0.25
w_delta = 0.05    #0.05
pg_delta = 0.25   #0.25

thermal_limits_model = true
voltage_angle_limits_model = true
cut_voltage_angle_limits_model = true
bound_slack_gen_model = true
bound_output_bus_voltage_model = true
bound_qg_model = true

##############################################################
#Run sweep over buses
##############################################################
result_dir = folder*"/constant_load_pf/violations_gurobi_pd_$(Int(pd_delta*100))_w_$(Int(w_delta*100))_pg_$(Int(pg_delta*100))"
mkpath(result_dir)

for i = 0:(config["runs"]-1) #for each sparsity run
    prefixp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(ld_int)_ldp_$(ld_prog)_run_$(i)"
    prefixnp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(epochs)_ldp_$(ld_prog)_run_$(i)"
    prefixes = [prefixp, prefixnp]
    for prefix in prefixes
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
            pg_delta = pg_delta,pd_delta = pd_delta,w_delta = w_delta,
            load_power_factor = :constant,
            gen_power_factor_model = :minimum,
            qd_limits = false,
            bound_slack_gen_model = bound_slack_gen_model,
            bound_qg_model = bound_qg_model,
            bound_output_bus_voltage_model = bound_output_bus_voltage_model,
            bound_c_s_model = true,
            thermal_limits_model = thermal_limits_model,
            voltage_angle_limits_model = voltage_angle_limits_model,
            cut_voltage_angle_limits_model = cut_voltage_angle_limits_model,
            gen_power_factor_nn = :off,
            bound_slack_gen_nn = false,
            bound_qg_nn = false,
            bound_output_bus_voltage_nn = false,
            bound_c_s_nn = false,
            thermal_limits_nn = false,
            voltage_angle_limits_nn = false,
            cut_voltage_angle_limits_nn = false)

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
            pg_delta = pg_delta,pd_delta = pd_delta,w_delta = w_delta,
            load_power_factor = :constant,
            gen_power_factor_model = :minimum,
            qd_limits = false,
            bound_slack_gen_model = bound_slack_gen_model,
            bound_qg_model = bound_qg_model,
            bound_output_bus_voltage_model = bound_output_bus_voltage_model,
            bound_c_s_model = true,
            thermal_limits_model = thermal_limits_model,
            voltage_angle_limits_model = voltage_angle_limits_model,
            cut_voltage_angle_limits_model = cut_voltage_angle_limits_model,
            gen_power_factor_nn = :off,
            bound_slack_gen_nn = false,
            bound_qg_nn = false,
            bound_output_bus_voltage_nn = false,
            bound_c_s_nn = false,
            thermal_limits_nn = false,
            voltage_angle_limits_nn = false,
            cut_voltage_angle_limits_nn = false)

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

        open(result_dir*"/$(prefix)_bus_errors.json","w") do f
            write(f, json_string)
        end
    end
end

#################################################
#Run the MAE verification problem
#################################################
for i = 0:(config["runs"]-1) #for each sparsity run
    prefixp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(ld_int)_ldp_$(ld_prog)_run_$(i)"
    prefixnp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(epochs)_ldp_$(ld_prog)_run_$(i)"
    prefixes = [prefixp, prefixnp]    
    for prefix in prefixes
        net_file = folder*"/$(prefix)_model_data.json"
        chain = read_acpf_json(net_file)

        #p mae
        model = build_acpf_wcs_verifier(dref,chain;nn_mode = NeuralOpt.IndicatorReluMode(),
        pg_delta = pg_delta,pd_delta = pd_delta,w_delta = w_delta,
        load_power_factor = :constant,
        gen_power_factor_model = :minimum,
        qd_limits = false,
        bound_slack_gen_model = bound_slack_gen_model,
        bound_qg_model = bound_qg_model,
        bound_output_bus_voltage_model = bound_output_bus_voltage_model,
        bound_c_s_model = true,
        thermal_limits_model = thermal_limits_model,
        voltage_angle_limits_model = voltage_angle_limits_model,
        cut_voltage_angle_limits_model = cut_voltage_angle_limits_model,
        gen_power_factor_nn = :off,
        bound_slack_gen_nn = false,
        bound_qg_nn = false,
        bound_output_bus_voltage_nn = false,
        bound_c_s_nn = false,
        thermal_limits_nn = false,
        voltage_angle_limits_nn = false,
        cut_voltage_angle_limits_nn = false)

        #gurobi = Gurobi.Optimizer
        #set_optimizer(model,gurobi)
        set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))
        set_optimizer_attribute(model,"NonConvex",2)
        set_optimizer_attribute(model,"PSDTol",1e-6)
        set_optimizer_attribute(model,"Threads",16)
        #set_optimizer_attribute(model,"Threads",1)
        set_optimizer_attribute(model,"TimeLimit",9600)

        # Indicator approach
        @variable(model,y_abs[i=1:n_buses])
        @variable(model,z1[1:n_buses],Bin)
        @constraint(model,indicator_max_1[i = 1:n_buses],!z1[i] => {y_abs[i] == model[:p_res][i]})  #z1 = 0 -> max p_res is positive
        @constraint(model,indicator_max_2[i = 1:n_buses], z1[i] => {y_abs[i] == -model[:p_res][i]})
        @objective(model,Max,sum(y_abs)/n_buses)
        optimize!(model)

        status_p_res = termination_status(model)
        p_res_mae = sum([abs(value(model[:p_res][i])) for i = 1:n_buses])/n_buses
        x_input_p_res = value.(model[:x])
        y_output_p_res = value.(model[:y])
        p_bound = objective_bound(model)
        solution_time_p_res = JuMP.solve_time(model)

        #q mae
        model = build_acpf_wcs_verifier(dref,chain;nn_mode = NeuralOpt.IndicatorReluMode(),
        pg_delta = pg_delta,pd_delta = pd_delta,w_delta = w_delta,
        load_power_factor = :constant,
        gen_power_factor_model = :minimum,
        qd_limits = false,
        bound_slack_gen_model = bound_slack_gen_model,
        bound_qg_model = bound_qg_model,
        bound_output_bus_voltage_model = bound_output_bus_voltage_model,
        bound_c_s_model = true,
        thermal_limits_model = thermal_limits_model,
        voltage_angle_limits_model = voltage_angle_limits_model,
        cut_voltage_angle_limits_model = cut_voltage_angle_limits_model,
        gen_power_factor_nn = :off,
        bound_slack_gen_nn = false,
        bound_qg_nn = false,
        bound_output_bus_voltage_nn = false,
        bound_c_s_nn = false,
        thermal_limits_nn = false,
        voltage_angle_limits_nn = false,
        cut_voltage_angle_limits_nn = false)

        #gurobi = Gurobi.Optimizer
        #set_optimizer(model,gurobi)
        set_optimizer(model, () -> Gurobi.Optimizer(GRB_ENV))
        set_optimizer_attribute(model,"NonConvex",2)
        set_optimizer_attribute(model,"PSDTol",1e-6)
        set_optimizer_attribute(model,"Threads",16)
        #set_optimizer_attribute(model,"Threads",1)
        set_optimizer_attribute(model,"TimeLimit",9600)

        # Indicator approach
        @variable(model,y_abs[i=1:n_buses])#,start = value(model_ipopt[:p_res][i]))
        @variable(model,z1[1:n_buses],Bin)
        @constraint(model,indicator_max_1[i = 1:n_buses],!z1[i] => {y_abs[i] == model[:q_res][i]})  #z1 = 0 -> max p_res is positive
        @constraint(model,indicator_max_2[i = 1:n_buses], z1[i] => {y_abs[i] == -model[:q_res][i]})
        @objective(model,Max,sum(y_abs)/n_buses)
        optimize!(model)

        status_q_res = termination_status(model)
        q_res_mae = sum([abs(value(model[:q_res][i])) for i = 1:n_buses])/n_buses
        x_input_q_res = value.(model[:x])
        y_output_q_res = value.(model[:y])
        q_bound = objective_bound(model)
        solution_time_q_res = JuMP.solve_time(model)

        mae_results = Dict("p_res_mae" => p_res_mae,"q_res_mae" => q_res_mae,"x_input_p_res" => x_input_p_res,"y_output_p_res" => y_output_p_res,
        "x_input_q_res" => x_input_q_res,"y_output_q_res" => y_output_q_res,"p_res_status" => status_p_res,"q_res_status" => status_q_res,
        "p_solve_time" => solution_time_p_res,"q_solve_time"=> solution_time_q_res)
        json_string = JSON.json(mae_results)

        open(result_dir*"/$(prefix)_bus_errors_mae.json","w") do f
            write(f, json_string)
        end
    end
end