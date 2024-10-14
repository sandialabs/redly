#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#Maximize output violation between neural net and a relaxed ACPF model.
using Gurobi
const GRB_ENV = Gurobi.Env()

using Formatting
using JSON, CSV
using DataFrames, Statistics
#load models
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
        
        max_error = []
        best_bound = []
        x_input = []
        y_output = []
        status_codes = []
        solve_times = []
        
        n_outputs = size(chain[end].weight)[1]
        for j = 1:n_outputs
            println(j)
            model = build_acpf_wcs_verifier(dref,chain;nn_mode = NeuralOpt.IndicatorReluMode(),
            pg_delta = pg_delta,pd_delta = pd_delta,w_delta = w_delta,
            qd_limits = false,
            load_power_factor = :constant,
            gen_power_factor_model = :minimum,
            bound_slack_gen_model = true,
            bound_qg_model = true,
            bound_output_bus_voltage_model = true,
            bound_c_s_model = true,
            thermal_limits_model = true,
            voltage_angle_limits_model = true,
            cut_voltage_angle_limits_model = true,
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
            set_optimizer_attribute(model,"TimeLimit",3600)
        
            #outputs: pref,qg,w_out,c,s
            y_model = [model[:pg][slack_gen_index];model[:qg];model[:w][bus_out_indices];model[:c];model[:s]]
        
            @variable(model,y_abs)
            @variable(model,z,Bin)
            @constraint(model,indicator_max_1,!z => {y_abs == (y_model[j] - model[:y][j])})
            @constraint(model,indicator_max_2, z => {y_abs == -(y_model[j] - model[:y][j])})
            @objective(model,Max,y_abs)
            optimize!(model)
        
            x = value.(model[:x])
            y = value.(model[:y])
            status = termination_status(model)
        
            push!(status_codes,status)
            push!(solve_times,JuMP.solve_time(model))
            push!(max_error,objective_value(model))
            push!(best_bound,objective_bound(model))
            push!(x_input,x)
            push!(y_output,y)
        end
        
        max_results = Dict("max_error" => max_error,"x_input" => x_input,"y_output" => y_output,
        "status" => status_codes,"solve_times" => solve_times,"best_bound" => best_bound)
        json_string = JSON.json(max_results)
        
        open(result_dir*"/$(prefix)_output_errors.json","w") do f
            write(f, json_string)
        end
    end
end
