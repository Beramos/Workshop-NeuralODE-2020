ν(G, L, Es, p) = p[1]*(G*L-Es/p[2])/(L + p[3]*G)

function rate(C)
    G, L, Es = C
    p = (9.03f-2, 5.0f-2, 1.9f1)
    ν(G, L, Es, p)
end

function reaction(du, u, p, t)
    G, L, Es = u
    dG  = du[1] = - ν(G, L, Es, p)
    dL  = du[2] = - ν(G, L, Es, p)
    dEs = du[3] =   ν(G, L, Es, p)
    return du
end

function reaction!(du, u, p, t)
    du = reaction(du, u, p, t)
end

function generate_true_solution(u₀, tₑ; nPoints=20)
    p = (9.03f-2, 5.0f-2, 1.9f1)
    tspan = (0.0, tₑ)
    
    prob = ODEProblem(reaction!, u₀, tspan, p, saveat=0:tₑ/nPoints:tₑ)
    sol = solve(prob, Tsit5())
    G = [c[1] for c in sol.u]
    L = [c[2] for c in sol.u]
    Es = [c[3] for c in sol.u]
    time = sol.t

  return permutedims([G L Es]), time
end

function plot_reaction(data)
  C, time = data

  pl = plot(xlabel="Time (min)", ylabel="Concentration (mM)", 
      ylims=(-2, 1.2*maximum(C)),
      xlims=(-2, time[end]*1.01))
  scatter!(pl, time, C[1,:], label="Sucrose")
  scatter!(pl, time, C[2,:], label="Stearic acid")
  scatter!(pl, time, C[3,:], label="Sucr-Stea-Esther")
  return pl
end

function prediction_vs_data(data, pred)
    C, time = data
    
    pl = plot_reaction(data)
    lcs = pl.series_list
    plot!(pl, time, pred[1,:], label = "", lc=lcs[1][:linecolor])
    plot!(pl, time, pred[2,:], label = "",  lc=lcs[2][:linecolor])
    plot!(pl, time, pred[3,:], label = "",  lc=lcs[3][:linecolor])
    return pl
end