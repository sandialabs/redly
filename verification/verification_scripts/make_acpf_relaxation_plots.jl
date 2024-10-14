#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#Create bar plots that we use to compare pinn vs purely data maximum residuals on each bus
#These plots compare the error obtained from the verifier versus the true prediction error obtained by plugging the result in an ACPF solver 
using Plots, Measures, JSON, StatsPlots, LaTeXStrings, Formatting
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

f_pinn_relax = open(result_dir*"/$(prefixp)_output_errors.json","r")
results_pinn_relax = JSON.parse(f_pinn_relax)
close(f_pinn_relax)

f_pinn_true = open(result_dir*"/$(prefixp)_acpf_errors.json","r")
results_pinn_true = JSON.parse(f_pinn_true)
close(f_pinn_true)

f_data_relax = open(result_dir*"/$(prefixnp)_output_errors.json","r")
results_data_relax = JSON.parse(f_data_relax)
close(f_data_relax)

f_data_true = open(result_dir*"/$(prefixnp)_acpf_errors.json","r")
results_data_true = JSON.parse(f_data_true)
close(f_data_true)

pinn_true = results_pinn_true["acpf_results"]
data_true = results_data_true["acpf_results"]

pinn_relax_error = results_pinn_relax["max_error"]
data_relax_error = results_data_relax["max_error"]

pinn_true_error = [pinn_true[i]["true_error"] for i = 1:length(pinn_true)]
data_true_error = [data_true[i]["true_error"] for i = 1:length(data_true)]

pref_out_index,qg_out_indices,w_out_indices,c_out_indices,s_out_indices = map_wcs_outputs(dref)
labels = [L"p_{ref}"]
for i = 1:length(qg_out_indices)
    push!(labels,L"q_{g%$i}")
end
for i = 1:length(w_out_indices)
    push!(labels,L"w_{%$i}")
end
for i = 1:length(c_out_indices)
    push!(labels,L"c_{%$i}")
end
for i = 1:length(s_out_indices)
    push!(labels,L"s_{%$i}")
end

n_outputs = length(labels)
inds = reverse(sortperm(abs.(pinn_true_error)))
pinn_relax_plot = pinn_relax_error[inds]
push!(pinn_relax_plot,minimum(pinn_true_error))
plt_violation = Plots.bar(framestyle = :box,guidefontsize = 24,tickfontsize = 18,legendfontsize = 18,legend = :topright,orientation = :horizontal,
size = (1600,800),xrotation = -45,margin=5mm);
x_tick_loc = collect(1:n_outputs)
xticks = (x_tick_loc,labels[inds])
Plots.xticks!(plt_violation,xticks);
Plots.xlims!(plt_violation,0,55.5);
Plots.ylims!(plt_violation,1e-6,maximum(pinn_relax_error)*1.5);
Plots.bar!(plt_violation,abs.(pinn_relax_plot);label = "PINN Relaxation",alpha = 0.5,orientation = :vertical,color = :blue,yaxis = (:log10,(1e-6,Inf)));
Plots.bar!(plt_violation,abs.(pinn_true_error)[inds];label = "PINN True",alpha = 0.75,orientation = :vertical,color = :purple,yaxis = :log10);
Plots.xlabel!(plt_violation,"Neural Net Output");
Plots.ylabel!(plt_violation,"Worst-Case Prediction Error");

Plots.savefig(plt_violation,result_dir*"/$(prefixp)_relaxation_error_pinn.png")
Plots.savefig(plt_violation,result_dir*"/$(prefixp)_relaxation_error_pinn.pdf")


#plot comparison with data
data_true_plot = data_true_error[inds]
push!(data_true_plot,minimum(pinn_true_error))
plt_violation_compare = Plots.bar(framestyle = :box,guidefontsize = 24,tickfontsize = 18,legendfontsize = 18,legend = :topright,orientation = :horizontal,
size = (1600,800),xrotation = -45,margin=5mm);
x_tick_loc = collect(1:n_outputs)
xticks = (x_tick_loc,labels[inds])
Plots.xticks!(plt_violation_compare,xticks);
Plots.xlims!(plt_violation_compare,0,55.5);
Plots.ylims!(plt_violation_compare,1e-6,maximum(data_relax_error)*1.5);
# sp.attr[:yaxis][:extrema].emin = 1e-6
Plots.bar!(plt_violation_compare,abs.(data_true_plot);label = "Data True",alpha = 0.5,orientation = :vertical,color = :red,yaxis = (:log10));
Plots.bar!(plt_violation_compare,abs.(pinn_true_error)[inds];label = "PINN True",alpha = 0.5,orientation = :vertical,color = :purple,yaxis = :log10);
Plots.xlabel!(plt_violation_compare,"Neural Net Output");
Plots.ylabel!(plt_violation_compare,"Worst-Case Prediction Error");

Plots.savefig(plt_violation_compare,result_dir*"/$(prefixp)_true_compare.png")
Plots.savefig(plt_violation_compare,result_dir*"/$(prefixp)_true_compare.pdf")
