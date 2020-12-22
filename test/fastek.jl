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


@testset "$ALG"for ALG in (FastEK0, FastEK1)

    @testset "$probname" for (prob, probname) in [
        (remake_prob_with_jac(prob_ode_lotkavoltera), "lotkavolterra"),
        (remake_prob_with_jac(prob_ode_fitzhughnagumo), "fitzhughnagumo"),
    ]
        true_sol = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)

        @testset "Constant steps w/o dense output; order $q" for q in 3:2:7
            sol = solve(prob, ALG(order=q, smooth=false),
                        dense=false,
                        adaptive=false, dt=1e-3)
            @test sol.u[end] ≈ true_sol.u[end] rtol=1e-8
            @test sol.u ≈ true_sol.(sol.t) rtol=1e-8
        end

        @testset "Adaptive step w/o smoothing; order $q" for q in 3:2:5
            sol = solve(prob, ALG(order=q, smooth=false),
                        dense=false,
                        abstol=1e-6, reltol=1e-6)
            @test sol.u[end] ≈ true_sol.u[end] rtol=(ALG==FastEK0 ? 1e-5 : 1e-2)
            @test sol.u ≈ true_sol.(sol.t) rtol=(ALG==FastEK0 ? 1e-5 : 1e-2)
        end

        # t_eval = prob.tspan[1]:0.05:prob.tspan[end]
        # true_dense_vals = true_sol.(t_eval)
    end

end
