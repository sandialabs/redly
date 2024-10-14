#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#Create scatter plots that show residual error for each bus and the MAE solution
using Plots, Measures, JSON, StatsPlots, Formatting
using DataFrames, CSV
using PowerModels
include((@__DIR__)*("/../utils.jl"))
include((@__DIR__)*("/acpf_plots.jl"))

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

#############################################################
#PLOT DOT COMPARISONS

p_res_max_pinn_times = []
p_res_max_data_times = []
p_res_mae_pinn_times = []
p_res_mae_data_times = []

q_res_max_pinn_times = []
q_res_max_data_times = []
q_res_mae_pinn_times = []
q_res_mae_data_times = []

p_res_max_pinn_status = []
p_res_max_data_status = []
p_res_mae_pinn_status = []
p_res_mae_data_status = []

q_res_max_pinn_status = []
q_res_max_data_status = []
q_res_mae_pinn_status = []
q_res_mae_data_status = []

for i in 0:(config["runs"]-1)
    prefixp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(ld_int)_ldp_$(ld_prog)_run_$(i)"
    prefixnp = "$(name)_b_$(b_p)_a_$(alpha)_d_$(dropout)_ldi_$(epochs)_ldp_$(ld_prog)_run_$(i)"
    
    f_pinn = open(result_dir*"/$(prefixp)_bus_errors.json","r")
    results_pinn = JSON.parse(f_pinn)
    close(f_pinn)
    push!(p_res_max_pinn_times,results_pinn["p_solve_times"])
    push!(q_res_max_pinn_times,results_pinn["q_solve_times"])
    push!(p_res_max_pinn_status,results_pinn["p_res_status"])
    push!(q_res_max_pinn_status,results_pinn["q_res_status"])

    f_mae_pinn = open(result_dir*"/$(prefixp)_bus_errors_mae.json","r")
    results_mae_pinn = JSON.parse(f_mae_pinn)
    close(f_mae_pinn)
    push!(p_res_mae_pinn_times,results_mae_pinn["p_solve_time"])
    push!(q_res_mae_pinn_times,results_mae_pinn["q_solve_time"])
    push!(p_res_mae_pinn_status,results_mae_pinn["p_res_status"])
    push!(q_res_mae_pinn_status,results_mae_pinn["q_res_status"])

    f_data = open(result_dir*"/$(prefixnp)_bus_errors.json","r")
    results_data = JSON.parse(f_data)
    close(f_data)
    push!(p_res_max_data_times,results_data["p_solve_times"])
    push!(q_res_max_data_times,results_data["q_solve_times"])
    push!(p_res_max_data_status,results_data["p_res_status"])
    push!(q_res_max_data_status,results_data["q_res_status"])

    f_mae_data = open(result_dir*"/$(prefixnp)_bus_errors_mae.json","r")
    results_mae_data = JSON.parse(f_mae_data)
    close(f_mae_data)
    push!(p_res_mae_data_times,results_mae_data["p_solve_time"])
    push!(q_res_mae_data_times,results_mae_data["q_solve_time"])
    push!(p_res_mae_data_status,results_mae_data["p_res_status"])
    push!(q_res_mae_data_status,results_mae_data["q_res_status"])
end


n_runs = length(p_res_max_pinn_times)
x_tick_loc = collect(1:n_runs)

plt = plot(xticks = (x_tick_loc,["$i" for i in 1:n_runs]),framestyle = :box,xlabel = "Pruning Run", ylabel = "Solution Time [s]",
legend = :topright,size = (800,800),guidefontsize = 24,tickfontsize = 24,legendfontsize = 18)#,yaxis = :log10);

Plots.xlims!(plt,(0,n_runs+1))

x_coords = collect(1:n_runs)
pinn_coords = x_coords .- 0.25
data_coords = x_coords .+ 0.25
separators1 = x_coords .+ 0.5
separators2 = x_coords .- 0.5

for i = 1:n_runs
    if i == 1
        scatter!(plt,[-100,-100],label = "PINN maximums",color = :blue)
    end
    y = [p_res_max_pinn_times[i];q_res_max_pinn_times[i]]
    x = pinn_coords[i]*ones(length(y))
    scatter!(plt,x,y,color = :blue,label = nothing,yaxis = :log10)
end

for i = 1:n_runs
    if i == 1
        scatter!(plt,[-100,-100],label = "Data-Only maximums",color = :red)
    end
    y = [p_res_max_data_times[i];q_res_max_data_times[i]]
    x = data_coords[i]*ones(length(y))
    scatter!(plt,x,y,color = :red,label = nothing,yaxis = :log10);
end

max_y = maximum([maximum.(p_res_max_pinn_times);maximum.(q_res_max_pinn_times);maximum.(q_res_max_data_times);maximum.(q_res_max_data_times)])
min_y = minimum([minimum.(p_res_max_pinn_times);minimum.(q_res_max_pinn_times);minimum.(q_res_max_data_times);minimum.(q_res_max_data_times)])

Plots.ylims!(plt,(min_y,max_y*1.1));
Plots.vline!(plt,separators1,color = :grey,label = nothing);
Plots.vline!(plt,separators2,color = :grey,label = nothing);
Plots.savefig(plt,result_dir*"/$(n_bus)_bus_sparsity_timing.png")
Plots.savefig(plt,result_dir*"/$(n_bus)_bus_sparsity_timing.pdf")
