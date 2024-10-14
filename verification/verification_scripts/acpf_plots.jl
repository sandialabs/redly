#  ___________________________________________________________________________
#
#  REDLY:  Resilience Enhancements for Deep Learning Yields
#  Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

#This file defines function to create some of the manuscript plots for pinn verification
using Plots

#Plot the maximum residuals for multiple sparsity runs.  Also acepts the mae solution for both pinn and data results
function plot_residuals_dots!(plt,max_pinn_array,max_data_array;annotate = false,mae_pinn = nothing,mae_data = nothing)
    @assert length(max_pinn_array) == length(max_data_array)
    n_runs = length(max_pinn_array)

    x_coords = collect(1:n_runs)
    pinn_coords = x_coords .- 0.25
    data_coords = x_coords .+ 0.25

    separators1 = x_coords .+ 0.5
    separators2 = x_coords .- 0.5
    #plt = plot(xticks = (x_tick_loc,["$i" for i = 0:n_runs - 1]),framestyle = :box,xlabel = "Sparsity Run", ylabel = "maximum bus p violations [p.u.]",legend = :topleft);
    if annotate == true
        alpha = 0
    else
        alpha = 1
    end


    for i = 1:length(max_pinn_array)
        if i == 1 && annotate == false
            scatter!(plt,[-100,-100],label = "pinn maximums",color = :blue)
        end
        y = max_pinn_array[i]
        x = pinn_coords[i]*ones(length(y))
        scatter!(plt,x,y,color = :blue,alpha = alpha,label = nothing)
        if annotate
            for j = 1:length(x)
                annotate!(plt,[(x[j], y[j], Plots.text("$j", 10, :blue, :center))]);
            end
        end
    end
    if !(mae_pinn == nothing)
        x = pinn_coords
        scatter!(plt,x,mae_pinn,color = :blue,marker = :diamond,markersize = 8,label = "PINN mae");
    end
    for i = 1:length(max_data_array)
        if i == 1 && annotate == false
            scatter!(plt,[-100,-100],label = "data maximums",color = :red)
        end
        y = max_data_array[i]
        x = data_coords[i]*ones(length(y))
        scatter!(plt,x,y,color = :red,label = nothing,alpha = alpha);
        if annotate
            for j = 1:length(x)
                annotate!(plt,[(x[j], y[j], Plots.text("$j", 10, :red, :center))]);
            end
        end
    end
    if !(mae_pinn == nothing)
        x = data_coords
        scatter!(plt,x,mae_data,color = :red,marker = :diamond,markersize = 8,label = "Data-Only mae");
    end

    max_y = maximum([maximum.(max_pinn_array);maximum.(max_data_array)])

    Plots.xlims!(plt,(0,n_runs+1))
    Plots.ylims!(plt,(-0.1,max_y*1.1))
    Plots.vline!(plt,separators1,color = :grey,label = nothing);
    Plots.vline!(plt,separators2,color = :grey,label = nothing);

    return plt
end