#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

# Use input solution to calculate true ACPF solution.  Compare outputs and branch flows. Write results to file.
using Formatting
using PowerModels
using DataFrames
#using MKL
using Ipopt

include((@__DIR__)*"/acpf_verifiers.jl")
include((@__DIR__)*"/../json_utils.jl")
include((@__DIR__)*("/../utils.jl"))

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

result_dir = folder*"/constant_load_pf/violations_gurobi_pd_25_w_5_pg_25"

pref_out_index,qg_out_indices,w_out_indices,c_out_indices,s_out_indices = map_wcs_outputs(dref)
bus_loads,load_to_bus,bus_gens,gen_to_bus,bus_shunts,shunt_to_bus,bus_arcs,arc_to_bus = load_mappings(dref)
slack_gen_index = get_slack_gen_index(dref)
bus_in_indices,bus_out_indices = get_in_out_bus_indices(dref)
g,b,tr,ti,g_fr,b_fr,g_to,b_to,tm,bus_fr,bus_to = load_branch_data(dref)
bus_pairs,branch_to_buspair = get_buspairs(dref)
bus_pairs = sort(bus_pairs) #NOTE: model assumes buspairs are sorted
n_buses = length(dref[:bus])
n_branches = length(dref[:branch])

for run = 0:(config["runs"]-1)
    prefixp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(ld_int)_ldp_$(ld_prog)_run_$(run)"
    prefixnp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(epochs)_ldp_$(ld_prog)_run_$(run)"
    prefixes = [prefixp, prefixnp]
    for prefix in prefixes
        result_file = result_dir*"/$(prefix)_output_errors.json"
        
        f = open(result_file,"r")
        arr = JSON.parse(f)
        close(f)

        inputs = arr["x_input"]
        outputs = arr["y_output"]

        #Unpack neural net outputs to ACPF results
        pg_in_indices,pd_in_indices,qd_in_indices,w_in_indices =  map_wcs_inputs(dref)
        pref_out_index,qg_out_indices,w_out_indices,c_out_indices,s_out_indices = map_wcs_outputs(dref)

        n_outputs = length(outputs)

        results = []
        for i = 1:n_outputs
            input = inputs[i]
            model_acpf = build_acpf_input_verifier(dref,input)            
            #ipopt = optimizer_with_attributes(Ipopt.Optimizer,"linear_solver" => "ma27")
            ipopt = Ipopt.Optimizer
            set_optimizer(model_acpf,ipopt)
            optimize!(model_acpf)

            pref_acpf = value(model_acpf[:pg][slack_gen_index])
            qg_acpf = value.(model_acpf[:qg])
            w_out_acpf = value.(model_acpf[:vm][bus_out_indices]).^2

            c_out_acpf = []
            s_out_acpf = []
            for bus_pair in bus_pairs
                bus_fr = bus_pair[1]
                bus_to = bus_pair[2]
                vm_fr = value(model_acpf[:vm][bus_fr])
                vm_to = value(model_acpf[:vm][bus_to])
                va_fr = value(model_acpf[:va][bus_fr])
                va_to = value(model_acpf[:va][bus_to])
                c = vm_fr*vm_to*cos(va_fr - va_to)
                s = vm_fr*vm_to*sin(va_fr - va_to)
                push!(c_out_acpf,c)
                push!(s_out_acpf,s)
            end

            model_outputs =  [pref_acpf;qg_acpf;w_out_acpf;c_out_acpf;s_out_acpf]
            nn_outputs = outputs[i]

            maxed_nn_output = nn_outputs[i]
            model_output = model_outputs[i]

            true_error = abs(maxed_nn_output - model_output)

            result = Dict("true_outputs" => model_outputs,"nn_outputs"=>nn_outputs,"true_error" => true_error)
            push!(results,result)
        end

        #Store results into another json file
        acpf_verification_file = result_dir*"/$(prefix)_acpf_errors.json"
        acpf_results = Dict("acpf_results" => results)
        json_string = JSON.json(acpf_results)
        open(acpf_verification_file,"w") do f
            write(f, json_string)
        end
    end
end
