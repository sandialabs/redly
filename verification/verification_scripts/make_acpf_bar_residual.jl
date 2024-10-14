#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#Create bar plots that show residual error for each bus and the MAE solution
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

# pg = 25
# w = 5
# pd = 25

#verification solutions for each bus
result_dir = folder*"/constant_load_pf/violations_gurobi_pd_25_w_5_pg_25"

#load solution results from json files
f_pinn = open(result_dir*"/$(prefixp)_bus_errors.json","r")
results_pinn = JSON.parse(f_pinn)
close(f_pinn)

f_data = open(result_dir*"/$(prefixnp)_bus_errors.json","r")
results_data = JSON.parse(f_data)
close(f_data)

f_pinn_mae = open(result_dir*"/$(prefixp)_bus_errors_mae.json","r")
results_pinn_mae = JSON.parse(f_pinn_mae)
close(f_pinn_mae)

f_data_mae = open(result_dir*"/$(prefixnp)_bus_errors_mae.json","r")
results_data_mae = JSON.parse(f_data_mae)
close(f_data_mae)

#Create the bar plots in the manuscript for this value of b
p_res_max_pinn = results_pinn["p_res_max"]
p_res_max_data = results_data["p_res_max"]
q_res_max_pinn = results_pinn["q_res_max"]
q_res_max_data = results_data["q_res_max"]

p_res_mae_pinn = results_pinn_mae["p_res_mae"]
p_res_mae_data = results_data_mae["p_res_mae"]
q_res_mae_pinn = results_pinn_mae["q_res_mae"]
q_res_mae_data = results_data_mae["q_res_mae"]

inds = reverse(sortperm(abs.(p_res_max_data)))
labels = ["$i" for i in inds]
plt_p_violation = Plots.bar(framestyle = :box,guidefontsize = 24,tickfontsize = 24,legendfontsize = 18,legend = :topright,orientation = :horizontal,
size = (800,800));
Plots.bar!(plt_p_violation,abs.(p_res_max_data)[inds];label = "Data-Only maximum",alpha = 0.75,orientation = :vertical,color = :red);
Plots.bar!(plt_p_violation,abs.(p_res_max_pinn)[inds];label = "PINN maximum",alpha = 0.75,orientation = :vertical,color = :blue);
Plots.hline!(plt_p_violation,[p_res_mae_data],label = "Data-Only mae", color = :red, linewidth = 2)
Plots.hline!(plt_p_violation,[p_res_mae_pinn],label = "PINN mae", color = :blue, linewidth = 2)

Plots.xlabel!(plt_p_violation,"Bus ID");
Plots.ylabel!(plt_p_violation,"Active Power Flow Violation [p.u.]");
x_tick_loc = collect(1:n_bus)
xticks = (x_tick_loc,labels)
Plots.xticks!(plt_p_violation,xticks);
Plots.savefig(plt_p_violation,result_dir*"/$(prefixp)_compare_p.png")
Plots.savefig(plt_p_violation,result_dir*"/$(prefixp)_compare_p.pdf")

inds = reverse(sortperm(abs.(q_res_max_data)))
labels = ["$i" for i in inds]
plt_q_violation = Plots.bar(framestyle = :box,guidefontsize = 24,tickfontsize = 24,legendfontsize = 18,legend = :topright,orientation = :horizontal,
size = (800,800));
Plots.bar!(plt_q_violation,abs.(q_res_max_data)[inds];label = "Data-Only maximum",alpha = 0.75,orientation = :vertical, color = :red);
Plots.bar!(plt_q_violation,abs.(q_res_max_pinn)[inds];label = "PINN maximum",alpha = 0.75,orientation = :vertical, color = :blue);
Plots.hline!(plt_q_violation,[q_res_mae_data],label = "Data-Only mae", color = :red, linewidth = 2)
Plots.hline!(plt_q_violation,[q_res_mae_pinn],label = "PINN mae", color = :blue, linewidth = 2)

Plots.xlabel!(plt_q_violation,"Bus ID");
Plots.ylabel!(plt_q_violation,"Reactive Power Flow Violation [p.u.]");
x_tick_loc = collect(1:n_bus)
xticks = (x_tick_loc,labels)
Plots.xticks!(plt_q_violation,xticks);
Plots.savefig(plt_q_violation,result_dir*"/$(prefixp)_compare_q.png")
Plots.savefig(plt_q_violation,result_dir*"/$(prefixp)_compare_q.pdf")
