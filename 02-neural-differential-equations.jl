### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ b72136c6-1302-11eb-0adc-a13323901a2e
begin
	import Pkg; Pkg.activate(".")
	using OrdinaryDiffEq, Plots
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
		
		return G, L, Es, time
	end
	
	function plot_reaction(data)
		G, L, Es, time = data
		pl = plot(xlabel="Time (min)", ylabel="Concentration (mM)", 
				ylims=(-2, 1.2e3*max(maximum(L),maximum(G),maximum(Es))),
			xlims=(-2, time[end]*1.01))
		scatter!(pl, time, G.*1e3, label="Glucose")
		scatter!(pl, time, L.*1e3, label="Lauric acid")
		scatter!(pl, time, Es.*1e3, label="Gluc-Lau-Esther")
	end
end

# ╔═╡ 33ba0d82-130a-11eb-1db7-578bc8ad56a3
data = generate_true_solution([45e-3 60e-3 0.0], 240.0)

# ╔═╡ 90ffaf22-130c-11eb-1563-b9d7cfa357bb
plot_reaction(data)

# ╔═╡ 3e9ee296-130e-11eb-2a97-b5c59c9ceffe


# ╔═╡ Cell order:
# ╠═b504da48-1300-11eb-3f13-87f09d20da82
# ╠═d0c166c8-1300-11eb-0084-7f6b803160ee
# ╠═d6bb8b50-1300-11eb-22d1-15d9f0c009b8
# ╟─fb9f5cee-1300-11eb-006c-5312a54e70f7
# ╠═b72136c6-1302-11eb-0adc-a13323901a2e
# ╠═944b2fe0-1301-11eb-1275-7b9c7af8cef1
# ╠═33ba0d82-130a-11eb-1db7-578bc8ad56a3
# ╠═90ffaf22-130c-11eb-1563-b9d7cfa357bb
# ╠═3e9ee296-130e-11eb-2a97-b5c59c9ceffe
