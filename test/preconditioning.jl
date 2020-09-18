using Test
using LinearAlgebra
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo


prob = prob_ode_lotkavoltera
prob = remake(prob, tspan=(0.0, 10.0))
prob = ProbNumODE.remake_prob_with_jac(prob)


@testset "Condition numbers of A,Q" begin
    h = 0.1*rand()
    σ = rand()

    d, q = 2, 3

    A!, Q! = ProbNumODE.vanilla_ibm(d, q)
    Ah = diagm(0 => ones(d*(q+1)))
    Qh = zeros(d*(q+1), d*(q+1))
    A!(Ah, h)
    Q!(Qh, h, σ^2)

    A!_p, Q!_p = ProbNumODE.ibm(d, q)
    Ah_p = diagm(0 => ones(d*(q+1)))
    Qh_p = zeros(d*(q+1), d*(q+1))
    A!_p(Ah_p, h)
    Q!_p(Qh_p, h, σ^2)


    # First test that they're both equivalent
    P, PI = ProbNumODE.preconditioner(d, q)
    P, PI = P(h), PI(h)
    @test Qh_p ≈ P * Qh * P'
    @test Ah_p ≈ P * Ah * PI

    # Check that the preconditioning actually helps
    @info "Condition numbers" cond(Qh) cond(Qh_p) cond(Ah) cond(Ah_p)
    @test cond(Qh) > cond(Qh_p)
    @test cond(Qh) > cond(Qh_p)^2
end
