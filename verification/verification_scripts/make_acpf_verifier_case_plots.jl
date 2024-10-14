#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#Create scatter plots that show residual error for each bus and the MAE solution for different verifier settings
using Plots, Measures, JSON, StatsPlots, Formatting
using DataFrames, CSV
using PowerModels
include((@__DIR__)*("/../utils.jl"))
include((@__DIR__)*("/acpf_plots.jl"))

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

result_dir = folder*"/constant_load_pf/violations_gurobi_pd_25_w_5_pg_25"
#############################################################
#PLOT DOT COMPARISONS
p_res_max_pinn_array = []
p_res_max_data_array = []
q_res_max_pinn_array = []
q_res_max_data_array = []

p_res_mae_pinn_array = []
p_res_mae_data_array = []
q_res_mae_pinn_array = []
q_res_mae_data_array = []

cases = [1,2,3,4]
for i in cases    
    f_pinn = open(result_dir*"/$(prefixp)_bus_errors_mae_case_$(i).json","r")
    results_pinn = JSON.parse(f_pinn)
    close(f_pinn)
    p_res_max_pinn = results_pinn["p_res_max"]
    q_res_max_pinn = results_pinn["q_res_max"]
    push!(p_res_max_pinn_array,abs.(p_res_max_pinn))
    push!(q_res_max_pinn_array,abs.(q_res_max_pinn))

    f_data = open(result_dir*"/$(prefixnp)_bus_errors_mae_case_$(i).json","r")
    results_data = JSON.parse(f_data)
    close(f_data)
    p_res_max_data = abs.(results_data["p_res_max"])
    q_res_max_data = abs.(results_data["q_res_max"])
    push!(p_res_max_data_array,p_res_max_data)
    push!(q_res_max_data_array,q_res_max_data)
end
f_mae_pinn = open(result_dir*"/$(prefixp)_bus_errors_mae_allcases.json","r")
results_mae_pinn = JSON.parse(f_mae_pinn)
close(f_mae_pinn)
p_res_mae_pinn_array = results_mae_pinn["p_res_mae"]
q_res_mae_pinn_array = results_mae_pinn["q_res_mae"]

f_mae_data = open(result_dir*"/$(prefixnp)_bus_errors_mae_allcases.json","r")
results_mae_data = JSON.parse(f_mae_data)
close(f_mae_data)
p_res_mae_data_array = results_mae_data["p_res_mae"]
q_res_mae_data_array = results_mae_data["q_res_mae"]

n_runs = length(cases)
x_tick_loc = collect(1:n_runs)

plt2 = plot(xticks = (x_tick_loc,["Case $i" for i = 1:n_runs]),framestyle = :box,xlabel = "Verifier Settings", ylabel = "Active Power Violations [p.u.]",
legend = :topright,size = (800,800),guidefontsize = 24,tickfontsize = 24,legendfontsize = 18);
plt2 = plot_residuals_dots!(plt2,p_res_max_pinn_array,p_res_max_data_array;annotate = true,mae_pinn = p_res_mae_pinn_array,mae_data = p_res_mae_data_array)
Plots.savefig(plt2,result_dir*"/$(n_bus)_bus_active_power_cases_verification.png")
Plots.savefig(plt2,result_dir*"/$(n_bus)_bus_active_power_cases_verification.pdf")

plt4 = plot(xticks = (x_tick_loc,["Case $i" for i = 1:n_runs]),framestyle = :box,xlabel = "Verifier Settings", ylabel = "Reactive Power Violations [p.u.]",
legend = :topright,size = (800,800),guidefontsize = 24,tickfontsize = 24,legendfontsize = 18);
plt4 = plot_residuals_dots!(plt4,q_res_max_pinn_array,q_res_max_data_array;annotate = true,mae_pinn = q_res_mae_pinn_array,mae_data = q_res_mae_data_array)
Plots.savefig(plt4,result_dir*"/$(n_bus)_bus_reactive_power_cases_verification.png")
Plots.savefig(plt4,result_dir*"/$(n_bus)_bus_reactive_power_cases_verification.pdf")
