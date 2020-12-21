# Goal: Make sure that our solvers are "correct" for small steps or tolerances
# Verify this for many (ideally all) combinations
# Compare with an algorithm from OrdinaryDiffEq.jl with high precision
using ODEFilters
using Test
using OrdinaryDiffEq
using LinearAlgebra
using Statistics: mean
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_lotkavoltera, prob_ode_fitzhughnagumo

import ODEFilters: remake_prob_with_jac


for (prob, probname) in [
    (remake_prob_with_jac(prob_ode_lotkavoltera), "lotkavolterra"),
    (remake_prob_with_jac(prob_ode_fitzhughnagumo), "fitzhughnagumo"),
]
    @testset "Constant steps: $probname" begin

        true_sol = solve(remake(prob, u0=big.(prob.u0)), Tsit5(), abstol=1e-20, reltol=1e-20)

        for Alg in (EK0, EK1),
            diffusion in [:fixed, :dynamic, :fixedMAP, :fixedMV, :dynamicMV],
            q in [1, 3, 5]

            if Alg == EK1 && diffusion in (:fixedMV, :dynamicMV) continue end

            @testset "Constant steps: $probname; q=$q, diffusion=$diffusion, alg=$Alg" begin

            @debug "Testing for correctness: Constant steps" probname alg diffusion q dt

            sol = solve(prob, Alg(order=q, diffusionmodel=diffusion),
                        adaptive=false, dt=5e-3)
            @test sol.u ≈ true_sol.(sol.t) rtol=1e-5
            end
        end
    end
end


for (prob, probname) in [
    (remake_prob_with_jac(prob_ode_lotkavoltera), "lotkavolterra"),
    (remake_prob_with_jac(prob_ode_fitzhughnagumo), "fitzhughnagumo"),
]
    @testset "Adaptive steps: $probname" begin

        t_eval = prob.tspan[1]:0.01:prob.tspan[end]
        true_sol = solve(remake(prob, u0=big.(prob.u0)), Tsit5(), abstol=1e-20, reltol=1e-20)
        true_dense_vals = true_sol.(t_eval)

        for Alg in (EK0, EK1),
            diffusion in [:fixed, :dynamic, :fixedMAP, :fixedMV, :dynamicMV],
            q in [2, 4, 6]

            if Alg == EK1 && diffusion in (:fixedMV, :dynamicMV) continue end

            @testset "Adaptive steps: $probname; q=$q, diffusion=$diffusion, Alg=$Alg" begin

            @debug "Testing for correctness: Adaptive steps" probname Alg diffusion q

            sol = solve(prob, Alg(order=q, diffusionmodel=diffusion),
                        adaptive=true)

            @test sol.u ≈ true_sol.(sol.t) rtol=1e-4
            @test mean.(sol.(t_eval)) ≈ true_dense_vals rtol=1e-4

            end
        end
    end
end
