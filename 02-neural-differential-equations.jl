### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ b72136c6-1302-11eb-0adc-a13323901a2e
begin
	import Pkg; Pkg.activate(".");
	using OrdinaryDiffEq, Plots, Optim
	using DiffEqFlux, Flux
end

# ╔═╡ b504da48-1300-11eb-3f13-87f09d20da82
md"# Neural differential equations"

# ╔═╡ d0c166c8-1300-11eb-0084-7f6b803160ee
md"**WIP uitleg**"

# ╔═╡ d6bb8b50-1300-11eb-22d1-15d9f0c009b8
md"""
## Biochemical reactor

$$\cfrac{dG}{dt} = -\nu(G, L, Es)$$
$$\cfrac{dL}{dt} = -\nu(G, L, Es)$$
$$\cfrac{dEs}{dt} =  \nu(G, L, Es)$$

"""

# ╔═╡ fb9f5cee-1300-11eb-006c-5312a54e70f7
md"**The dataset**"

# ╔═╡ 944b2fe0-1301-11eb-1275-7b9c7af8cef1
begin
	ν(G, L, Es, p) = p[1]*(G*L-Es/p[2])/(L + p[3]*G)
	
	function reaction!(du, u, p, t)
		G, L, Es = u
		dG  = du[1] = - ν(G, L, Es, p)
		dL  = du[2] = - ν(G, L, Es, p)
		dEs = du[3] =   ν(G, L, Es, p)
	end
	
	function generate_true_solution(u₀, tₑ; nPoints=20)
		p = (1.03e-2, 5.0, 1.9)
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
end

# ╔═╡ 33ba0d82-130a-11eb-1db7-578bc8ad56a3
data = generate_true_solution([45.0 60.0 0.0], 240.0)

# ╔═╡ 90ffaf22-130c-11eb-1563-b9d7cfa357bb
plot_reaction(data)

# ╔═╡ 3e9ee296-130e-11eb-2a97-b5c59c9ceffe
md"""### Training neural differential equations"""

# ╔═╡ 5a157c88-130e-11eb-210e-41767b91936f
dCdt = Chain(Dense(1,10, tanh), Dense(10,10, tanh), Dense(10,1))

# ╔═╡ 65c88844-13af-11eb-1a89-d3e756c0856f
tsteps = range(0.0, 240.0, length = 21)

# ╔═╡ ef373306-130e-11eb-2e52-05d3b9a93482
prob = NeuralODE(dCdt, (0.0, 240.0), Tsit5(), 
	saveat = tsteps);

# ╔═╡ ca347424-13ae-11eb-02dd-671342aa28e2
C, time = data[1][1,:], data[2];

# ╔═╡ b1be5ef2-13ac-11eb-3ca9-eb916d44574f
u₀ = [45]

# ╔═╡ 2ebc3a76-130f-11eb-0dc0-f1abda11b15c
function predict_neuralODE(p)
	Array(prob(u₀, p))
end

# ╔═╡ 7dce864e-13ac-11eb-21b0-45e74622be3b
predict_neuralODE(prob.p)

# ╔═╡ 4c4380e0-130f-11eb-1a10-bd40fbbf5bbb
L₁(p) = sum((C.-predict_neuralODE(p)).^2);

# ╔═╡ fdb0a068-13ae-11eb-197b-9d95c33c6610
callback = function (p, l; doplot = false)
  display(l)
  # plot current prediction against data
  pred = predict_neuralODE(p)
  pl = plot_reaction(data)
  plot!(pl, tsteps, pred[1,:], label = "prediction")
  if doplot
    display(plot(pl))
  end
  return false
end

# ╔═╡ cb4aae50-13b3-11eb-2e9d-293c3bcc12ef
begin
	pred =predict_neuralODE(prob.p)
	pl = plot_reaction(data)
	plot!(pl, tsteps, pred[1, :], label = "prediction")
end

# ╔═╡ 81af9234-13c4-11eb-1962-8bf0066edb14
#result_neuralode = DiffEqFlux.sciml_train(L₁, prob.p,
                                          ADAM(0.1), cb = callback,
                                          maxiters = 100)

# ╔═╡ f184d9f2-13c4-11eb-18fc-37ab6d416881
#result_neuralode2 = DiffEqFlux.sciml_train(L₁, result_neuralode.minimizer,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 200)

# ╔═╡ 0ad44e60-13c7-11eb-1246-d5b86ab65909
#result_neuralode3 = DiffEqFlux.sciml_train(L₁,
                                           result_neuralode2.minimizer,
                                           LBFGS(),
                                           cb = callback,
                                           allow_f_increases = false)

# ╔═╡ d669610c-13c5-11eb-383c-a592c63662e7
pFinal = result_neuralode3.minimizer;

# ╔═╡ 62f119ba-13c6-11eb-159b-19ff07ac4833
begin
	pred2 =predict_neuralODE(pFinal)[:]
	pl2 = plot_reaction(data)
	plot!(pl2, tsteps, 1e3*pred2, label = "prediction")
end

# ╔═╡ Cell order:
# ╠═b504da48-1300-11eb-3f13-87f09d20da82
# ╠═d0c166c8-1300-11eb-0084-7f6b803160ee
# ╠═d6bb8b50-1300-11eb-22d1-15d9f0c009b8
# ╟─fb9f5cee-1300-11eb-006c-5312a54e70f7
# ╠═b72136c6-1302-11eb-0adc-a13323901a2e
# ╠═944b2fe0-1301-11eb-1275-7b9c7af8cef1
# ╠═33ba0d82-130a-11eb-1db7-578bc8ad56a3
# ╠═90ffaf22-130c-11eb-1563-b9d7cfa357bb
# ╟─3e9ee296-130e-11eb-2a97-b5c59c9ceffe
# ╠═5a157c88-130e-11eb-210e-41767b91936f
# ╠═65c88844-13af-11eb-1a89-d3e756c0856f
# ╠═ef373306-130e-11eb-2e52-05d3b9a93482
# ╠═2ebc3a76-130f-11eb-0dc0-f1abda11b15c
# ╠═ca347424-13ae-11eb-02dd-671342aa28e2
# ╠═b1be5ef2-13ac-11eb-3ca9-eb916d44574f
# ╟─7dce864e-13ac-11eb-21b0-45e74622be3b
# ╠═4c4380e0-130f-11eb-1a10-bd40fbbf5bbb
# ╠═fdb0a068-13ae-11eb-197b-9d95c33c6610
# ╠═cb4aae50-13b3-11eb-2e9d-293c3bcc12ef
# ╠═81af9234-13c4-11eb-1962-8bf0066edb14
# ╠═f184d9f2-13c4-11eb-18fc-37ab6d416881
# ╠═0ad44e60-13c7-11eb-1246-d5b86ab65909
# ╠═d669610c-13c5-11eb-383c-a592c63662e7
# ╠═62f119ba-13c6-11eb-159b-19ff07ac4833
