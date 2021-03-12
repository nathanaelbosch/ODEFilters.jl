"""
    remake_prob_with_jac(prob)

Add a jacobian function to the ODE function, using ModelingToolkit.jl.
"""
function remake_prob_with_jac(prob::ODEProblem)
    IIP = isinplace(prob.f)
    try
        p = prob.p isa DiffEqBase.NullParameters ? [] : collect(prob.p)
        prob = remake(prob, p=p)
        sys = modelingtoolkitize(prob)
        jac = eval(ModelingToolkit.generate_jacobian(sys)[IIP ? 2 : 1])
        f = ODEFunction{IIP}(
            prob.f.f,
            jac=jac;
            analytic=prob.f.analytic,
        )
        return remake(prob, f=f)
    catch e
        @error "Could not generate a jacobian for the problem"
        throw(e)
    end
end
