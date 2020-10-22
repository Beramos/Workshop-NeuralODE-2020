using OrdinaryDiffEq, Plots, Optim
using DiffEqFlux, Flux

ν(G, L, Es, p) = p[1]*(G*L-Es/p[2])/(L + p[3]*G)

function reaction!(du, u, p, t)
  G, L, Es = u
  dG  = du[1] = - ν(G, L, Es, p)
  dL  = du[2] = - ν(G, L, Es, p)
  dEs = du[3] =   ν(G, L, Es, p)
end

function generate_true_solution(u₀, tₑ; nPoints=20)
  p = (1.03e-2, 5.0e-2, 1.9e-3)
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
  scatter!(pl, time, C[1,:], label="Glucose")
  scatter!(pl, time, C[2,:], label="Lauric acid")
  scatter!(pl, time, C[3,:], label="Gluc-Lau-Esther")
  return pl
end

data = generate_true_solution([45.0 60.0 0.0], 240.0)
plot_reaction(data)

dCdt = Chain(Dense(3, 8, σ), Dense(8, 8, σ), Dense(8,1), x->x.*[1.0; 1.0; -1.0])
tsteps = range(0.0, 240.0, length = 21)
prob = NeuralODE(dCdt, (0.0, 240.0), Tsit5(), 
  saveat = tsteps);
  
function predict_neuralODE(p)
  Array(prob(u₀, p))
end

L₁(p) = sum((C.-predict_neuralODE(p)).^2);

cb = function (p, l; doplot = true)
  display(l)
  # plot current prediction against data
  pred = predict_neuralODE(p)
  pl = plot_reaction(data)
  lcs = pl.series_list
  plot!(pl, tsteps, pred[1,:], label = "", lc=lcs[1][:linecolor])
  plot!(pl, tsteps, pred[2,:], label = "",  lc=lcs[2][:linecolor])
  plot!(pl, tsteps, pred[3,:], label = "",  lc=lcs[3][:linecolor])
  if doplot
    display(plot(pl))
  end
  return false
end
C = data[1]
u₀ = [45.0; 60.0; 0.0]

cb(prob.p, 0.0)
result_neuralode = DiffEqFlux.sciml_train(L₁, prob.p,
  ADAM(0.05), cb = cb,
  maxiters = 200)