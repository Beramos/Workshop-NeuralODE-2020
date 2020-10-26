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


p,re = Flux.destructure(dCdt)

trainedModel(u,p,t) = re(result_neuralode.minimizer)(u)

trainedModel([0.0, 0.0, 0.0],result_neuralode.minimizer, 0.0)

### Scaling up the reaction
function reactorNN!(du, u, p, t)
  G, L, Es = u
  Gᵢ, Lᵢ, Esᵢ = Cᵢ
  dG  = du[1] = 1/V*(Qᵢ*Gᵢ-Qᵢ*G) + trainedModel(u, p, t)[1]
  dL  = du[2] = 1/V*(Qᵢ*Lᵢ-Qᵢ*L) + trainedModel(u, p, t)[1]
  dEs = du[3] = 1/V*(Qᵢ*Esᵢ-Qᵢ*Es) - trainedModel(u, p, t)[1]
end

### Scaling up the reaction
function reactorNN!(du, u, p, t)
  G, L, Es = u
  Gᵢ, Lᵢ, Esᵢ = Cᵢ
  dG  = du[1] = 1/V*(Qᵢ*Gᵢ-Qᵢ*G) + trainedModel(u, p, t)[1]
  dL  = du[2] = 1/V*(Qᵢ*Lᵢ-Qᵢ*L) + trainedModel(u, p, t)[1]
  dEs = du[3] = 1/V*(Qᵢ*Esᵢ-Qᵢ*Es) - trainedModel(u, p, t)[1]
end

### Scaling up the reaction
function reactor!(du, u, p, t)
  p₂ = (1.03e-2, 5.0e-2, 1.9e-3)
  G, L, Es = u
  Gᵢ, Lᵢ, Esᵢ = Cᵢ

  dG  = du[1] = 1/V*(Qᵢ*Gᵢ-Qᵢ*G) - ν(G, L, Es, p₂)
  dL  = du[2] = 1/V*(Qᵢ*Lᵢ-Qᵢ*L) - ν(G, L, Es, p₂) 
  dEs = du[3] = 1/V*(Qᵢ*Esᵢ-Qᵢ*Es) + ν(G, L, Es, p₂)
end

V = 10.0
Qᵢ = 1.0
Cᵢ = u₀

probReactorNN = ODEProblem(reactorNN!, u₀, (0.0, 480), result_neuralode.minimizer)
solNN = solve(probReactorNN, Tsit5())

probReactor = ODEProblem(reactor!, u₀, (0.0, 480), result_neuralode.minimizer)
sol = solve(probReactor, Tsit5())

#reactor!([0.,0.,0.], [0.0, 0.0, 0.0], result_neuralode.minimizer, 0.0)

pl = plot()

G = [C[1] for C in sol.u]
L = [C[2] for C in sol.u]
Es = [C[3] for C in sol.u]

scatter!(pl, sol.t, G, label="")
scatter!(pl, sol.t, L, label="")
scatter!(pl, sol.t, Es, label="")

GNN = [C[1] for C in solNN.u]
LNN = [C[2] for C in solNN.u]
EsNN = [C[3] for C in solNN.u]

lcs = pl.series_list
plot!(pl, solNN.t, GNN, label = "", lc=lcs[1][:linecolor])
plot!(pl, solNN.t, LNN, label = "", lc=lcs[2][:linecolor])
plot!(pl, solNN.t, EsNN, label = "", lc=lcs[3][:linecolor])

EsNNLst = []
for Q in [1.5, 1.75, 2.0]

  global Qᵢ = Q
  probReactorNN = ODEProblem(reactorNN!, u₀, (0.0, 120.0), result_neuralode.minimizer)
  solNN = solve(probReactorNN, Tsit5(), saveat=1.0)

 
  push!(EsNNLst, [C[3] for C in solNN.u])
end

probReactor = ODEProblem(reactor!, u₀, (0.0, 120.0), result_neuralode.minimizer)
sol = solve(probReactor, Tsit5(), saveat=5.0)
Es = [C[3] for C in sol.u]


time = 0.0:1.0:120.0
pl = plot()
plot!(time, EsNNLst[1], label="Es (1.5)")
plot!(time, EsNNLst[2], label="Es (1.75)")
plot!(time, EsNNLst[3], label="Es (2.0)")

scatter!(sol.t, Es, label="Data", mc=:grey)
