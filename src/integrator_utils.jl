# Calibration, smoothing, then jump to the OrdinaryDiffEq._postamble!
function OrdinaryDiffEq.postamble!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK})

    if isstatic(integ.cache.diffusionmodel) # Calibrate
        # @warn "sol.log_likelihood is not correct for static diffusion models!"
        integ.sol.log_likelihood = NaN
        final_diff = integ.sol.diffusions[end]
        for s in integ.sol.x
            # s.Σ .*= final_diff
            copy!(s.Σ, apply_diffusion(s.Σ, final_diff))
        end

        if isempty(size(final_diff))
            integ.sol.diffusions .= final_diff
        else
            [(d .= final_diff) for d in integ.sol.diffusions]
        end
    end

    if integ.alg.smooth
        smooth_all!(integ)
        integ.sol.pu .= [integ.cache.SolProj * x for x in integ.sol.x]
        @assert (length(integ.sol.u) == length(integ.sol.pu))
        [(su .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
    end


    OrdinaryDiffEq._postamble!(integ)
end


function DiffEqBase.savevalues!(
    integrator::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK},
    force_save=false, reduce_size=true)

    @assert integrator.opts.dense
    @assert integrator.saveiter == integrator.saveiter_dense

    # Do whatever OrdinaryDiffEq would do
    out = OrdinaryDiffEq._savevalues!(integrator, force_save, reduce_size)

    # Save our custom stuff that we need for the posterior
    OrdinaryDiffEq.copyat_or_push!(integrator.sol.x, integrator.saveiter, integrator.cache.x)
    OrdinaryDiffEq.copyat_or_push!(integrator.sol.diffusions, integrator.saveiter, integrator.cache.diffusion)
    OrdinaryDiffEq.copyat_or_push!(integrator.sol.pu, integrator.saveiter, integrator.cache.SolProj*integrator.cache.x)

    integrator.sol.log_likelihood += integrator.cache.log_likelihood

    return out
end
