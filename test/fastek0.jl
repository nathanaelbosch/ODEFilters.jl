"""
Goal: Make sure that FastEK0 "runs" for a range of configurations and provides "reasonable" results

Configs of interest:
- Multiple orders
- With and without smoothing
- TODO Dense output
"""

using ODEFilters
using Test
using OrdinaryDiffEq
using LinearAlgebra
using Statistics: mean
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_lotkavoltera, prob_ode_fitzhughnagumo

import ODEFilters: remake_prob_with_jac


ALG = FastEK0


@testset "FastEK0 on $probname" for (prob, probname) in [
    (remake_prob_with_jac(prob_ode_lotkavoltera), "lotkavolterra"),
    (remake_prob_with_jac(prob_ode_fitzhughnagumo), "fitzhughnagumo"),
]
    true_sol = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)

    @testset "Constant steps $probname w/o dense output; order $q" for q in 3:2:7
        sol = solve(prob, ALG(order=q, smooth=false), adaptive=false, dt=1e-3, dense=false, save_everystep=false)
        @test sol.u[end] ≈ true_sol.u[end] rtol=1e-8
    end

    # t_eval = prob.tspan[1]:0.05:prob.tspan[end]
    # true_dense_vals = true_sol.(t_eval)
    @testset "Adaptive step with smoothing; order $q" for q in 3:2:7
        sol = solve(prob, ALG(order=q), abstol=1e-6, reltol=1e-3)
        @test sol.u[end] ≈ true_sol.u[end] rtol=1e-5
        # @test mean.(sol.(t_eval)) ≈ true_dense_vals rtol=1e-5
        @test sol.u ≈ true_sol.(sol.t) rtol=1e-5
    end
end
