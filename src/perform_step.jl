# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)
    @assert integ.opts.dense == integ.alg.smooth "`dense` and `smooth` should have the same value! "
    @assert integ.saveiter == 1
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj*cache.x)
end

"""Perform a step

Not necessarily successful! For that, see `step!(integ)`.

Basically consists of the following steps
- Coordinate change / Predonditioning
- Prediction step
- Measurement: Evaluate f and Jf; Build z, S, H
- Calibration; Adjust prediction / measurement covs if the diffusion model "dynamic"
- Update step
- Error estimation
- Undo the coordinate change / Predonditioning
"""
function OrdinaryDiffEq.perform_step!(integ, cache::GaussianODEFilterCache, repeat_step=false)
    @unpack t, dt = integ
    @unpack d, Proj, SolProj, Precond = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack A, Q = integ.cache

    tnew = t + dt

    # Coordinate change / preconditioning
    P = Precond(dt)
    PI = inv(P)
    x = P * x

    if isdynamic(cache.diffusionmodel)  # Calibrate, then predict cov

        # Predict
        predict_mean!(x_pred, x, A, Q)
        mul!(u_pred, SolProj, PI*x_pred.μ)

        # Measure
        measure!(integ, x_pred, tnew)

        # Estimate diffusion
        integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)
        # Adjust prediction and measurement
        predict_cov!(x_pred, x, A, apply_diffusion(Q, integ.cache.diffusion))
        copy!(integ.cache.measurement.Σ, Matrix(X_A_Xt(x_pred.Σ, integ.cache.H)))

    else  # Vanilla filtering order: Predict, measure, calibrate

        predict!(x_pred, x, A, Q)
        mul!(u_pred, SolProj, PI*x_pred.μ)
        measure!(integ, x_pred, tnew)
        integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)

    end

    # Likelihood
    # cache.log_likelihood = logpdf(cache.measurement, zeros(d))

    # Update
    x_filt = update!(integ, x_pred)
    mul!(u_filt, SolProj, PI*x_filt.μ)
    integ.u .= u_filt

    # Undo the coordinate change / preconditioning
    copy!(integ.cache.x, PI * x)
    copy!(integ.cache.x_pred, PI * x_pred)
    copy!(integ.cache.x_filt, PI * x_filt)

    # Estimate error for adaptive steps
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(integ, integ.cache)
        DiffEqBase.calculate_residuals!(
            err_tmp, dt * err_est_unscaled, integ.u, u_filt,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(err_tmp, t) # scalar

    end
    # stuff that would normally be in apply_step!
    if !integ.opts.adaptive || integ.EEst < one(integ.EEst)
        copy!(integ.cache.x, integ.cache.x_filt)
        integ.sol.log_likelihood += integ.cache.log_likelihood
    end
end


function h!(integ, x_pred, t)
    @unpack f, p, dt = integ
    @unpack u_pred, du, Proj, Precond, measurement = integ.cache
    PI = inv(Precond(dt))
    z = measurement.μ
    E0, E1 = Proj(0), Proj(1)

    u_pred .= E0*PI*x_pred.μ
    IIP = isinplace(integ.f)
    if IIP
        f(du, u_pred, p, t)
    else
        du .= f(u_pred, p, t)
    end
    integ.destats.nf += 1

    z .= f.mass_matrix*E1*PI*x_pred.μ .- du

    return z
end

function H!(integ, x_pred, t)
    @unpack f, p, dt, alg = integ
    @unpack ddu, Proj, Precond, H, u_pred = integ.cache
    E0, E1 = Proj(0), Proj(1)
    PI = inv(Precond(dt))

    if alg isa EK1 || alg isa IEKS
        if alg isa IEKS && !isnothing(alg.linearize_at)
            linearize_at = alg.linearize_at(t).μ
        else
            linearize_at = u_pred
        end

        if isinplace(integ.f)
            f.jac(ddu, linearize_at, p, t)
        else
            ddu .= f.jac(linearize_at, p, t)
            # WIP: Handle Jacobians as OrdinaryDiffEq.jl does
            # J = OrdinaryDiffEq.jacobian((u)-> f(u, p, t), u_pred, integ)
            # @assert J ≈ ddu
        end
        integ.destats.njacs += 1
        mul!(H, (f.mass_matrix*E1 .- ddu * E0), PI)
    else
        mul!(H, f.mass_matrix*E1, PI)
    end

    return H
end


function measure!(integ, x_pred, t)
    @unpack R = integ.cache
    @unpack u_pred, measurement, H = integ.cache

    z, S = measurement.μ, measurement.Σ
    z .= h!(integ, x_pred, t)
    H .= H!(integ, x_pred, t)
    # R .= Diagonal(eps.(z))
    @assert iszero(R)
    copy!(S, Matrix(X_A_Xt(x_pred.Σ, H)))

    return nothing
end


function update!(integ, prediction)
    @unpack measurement, H, R, x_filt = integ.cache
    update!(x_filt, prediction, measurement, H, R)
    # assert_nonnegative_diagonal(x_filt.Σ)
    return x_filt
end


function estimate_errors(integ, cache::GaussianODEFilterCache)
    if integ.alg.errest == :defect
        return defect_error_estimate(integ, cache)
    elseif integ.alg.errest == :embedded || integ.alg.errest == :cheap_embedded
        return embedded_error_estimate(integ, cache)
    # elseif integ.alg.errest == :richardson
    #     return richardson_error_estimate(integ, cache)
    else
        error("invalid errest: $(integ.alg.errest)")
    end
end

function defect_error_estimate(integ, cache)
    @unpack diffusion, Q, H = integ.cache

    if diffusion isa Real && isinf(diffusion)
        return Inf
    end

    error_estimate = sqrt.(diag(Matrix(X_A_Xt(apply_diffusion(Q, diffusion), H))))

    return error_estimate
end


function embedded_error_estimate(integ, cache)
    @unpack diffusion, Q, H = integ.cache

    if diffusion isa Real && isinf(diffusion)
        return Inf
    end

    # Try out some wild things on embedded error estimation
    @unpack t, dt = integ
    @unpack d, Proj, SolProj, Precond = integ.cache
    @unpack x, x_pred, x_filt, u_filt = integ.cache
    @unpack A, Q = integ.cache
    q = integ.alg.order
    D = d*(q+1)
    q_l = integ.alg.emb_order == 0 ? integ.alg.order-1 : integ.alg.emb_order
    D_l = d*(q_l+1)

    P = Precond(dt)
    PI = inv(P)
    # x = P*x
    # E0 = Proj(0)
    # Ah = PI * A * P
    # Qh = X_A_Xt(Q, PI)

    # Just to test: re-create the prediction step in here
    # x_tmp = copy(x_pred)
    # predict!(x_tmp, x, Ah, apply_diffusion(Qh, integ.cache.diffusion))
    # @assert x_tmp ≈ x_pred

    # Now do the thing with a lower order
    m_l = x.μ[1:D_l]
    P_l_L = collect(qr(x.Σ.squareroot[1:D_l, :]').R')
    P_l = SquarerootMatrix(P_l_L)
    x_l = Gaussian(m_l, P_l)

    A_l, Q_l = ibm(d, q_l)
    Precond_l = preconditioner(d, q_l)
    P_l = Precond_l(dt)
    PI_l = inv(P_l)
    Ah_l = PI_l * A_l * P_l
    Qh_l = X_A_Xt(Q_l, PI_l)

    # Predict mean
    x_pred_l = copy(x_l)
    predict_mean!(x_pred_l, x_l, Ah_l, Qh_l)
    # @info "predict with lower order" x_pred_l.μ x_pred.μ

    # measure
    @unpack f, p, dt, alg, t = integ
    @unpack u_pred, du, ddu, Proj, Precond, measurement, R, H = integ.cache
    E0 = Proj(0)[:, 1:D_l]
    E1 = Proj(1)[:, 1:D_l]
    _m = copy(measurement)
    z, S = _m.μ, _m.Σ
    if !(integ.alg isa DAE_EK1)
        if integ.alg.errest == :embedded
            _eval_f!(du, E0 * x_pred_l.μ, p, t+dt, f)
            integ.destats.nf += 1
        end
        z .= E1*x_pred_l.μ .- du
    else
        f(z, E1 * x_pred_l.μ, E0 * x_pred_l.μ, p, t+dt)
    end

    @assert !(alg isa IEKS)
    if !(integ.alg isa DAE_EK1)
        H = copy(H)[:, 1:D_l]
        if alg isa EK1
            # _eval_f_jac!(ddu, E0*x_pred_l.μ, p, t+dt, f)
            # integ.destats.njacs += 1
            #H .= E1 .- ddu * E0
            H .= (f.mass_matrix*E1 .- ddu * E0)
        else
            mul!(H, f.mass_matrix, E1)
        end
        copy!(S, Matrix(X_A_Xt(x_pred_l.Σ, H)))
    else
        @assert isinplace(integ.f)
        u_pred = E0*x_pred_l.μ
        du_pred = E1*x_pred_l.μ
        Ju = ForwardDiff.jacobian((u) -> (tmp = copy(u); f(tmp, du_pred, u, p, t); tmp), u_pred)
        Jdu = ForwardDiff.jacobian((du) -> (tmp = copy(du); f(tmp, du, u_pred, p, t); tmp), du_pred)
        H = (Jdu*E1 + Ju*E0)
        copy!(S, Matrix(X_A_Xt(x_pred_l.Σ, H)))
    end

    # Get diffusion
    diffusion_l = z' * (Matrix(X_A_Xt(Qh_l, H))\z) / d
    # Predict cov
    predict_cov!(x_pred_l, x_l, Ah_l, apply_diffusion(Qh_l, diffusion_l))
    # Adjust S
    copy!(S, Matrix(X_A_Xt(x_pred_l.Σ, H)))

    # update
    x_filt_l = copy(x_pred_l)
    update!(x_filt_l, x_pred_l, _m, H, R)



    # Finally: Compare the orders!
    # @info "Estimate_errors" E0*x_filt_l.μ - integ.cache.u_filt
    error_estimate = E0*x_filt_l.μ - integ.cache.u_filt


    # @info "estimate_errors" x.μ
    return error_estimate
end
