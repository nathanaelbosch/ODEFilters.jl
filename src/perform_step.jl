"""Perform a step, but not necessarily successful!

This is the actual interestin part of the algorithm
"""
function perform_step!(integ, cache::GaussianODEFilterCache)
    @unpack t, dt = integ
    @unpack E0, Precond, InvPrecond = integ.cache
    @unpack x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache

    P = Precond(dt)
    PI = InvPrecond(dt)
    integ.cache.x = P * integ.cache.x

    t = t + dt
    integ.t_new = t

    x_pred = predict!(integ)
    mul!(u_pred, E0, PI*x_pred.μ)

    measure!(integ, x_pred, t)

    if isdynamic(integ.sigma_estimator)
        σ_sq = dynamic_sigma_estimation(integ.sigma_estimator, integ)

        # Adjust prediction and measurement accordingly
        x_pred.Σ .+= (σ_sq .- 1) .* integ.cache.Qh
        integ.cache.measurement.Σ .+= integ.cache.H * ((σ_sq .- 1) .* integ.cache.Qh) * integ.cache.H'

        integ.cache.σ_sq = σ_sq
    end

    x_filt = update!(integ, x_pred)
    mul!(u_filt, E0, PI*x_filt.μ)

    if isstatic(integ.sigma_estimator)
        # E.g. estimate the /current/ MLE sigma; Needed for error estimation
        σ_sq = static_sigma_estimation(integ.sigma_estimator, integ)
        integ.cache.σ_sq = σ_sq
    end

    err_est_unscaled = estimate_errors(integ.error_estimator, integ)
    # Scale the error with old u-values and tolerances
    DiffEqBase.calculate_residuals!(
        err_tmp,
        dt * err_est_unscaled, integ.u, u_filt, integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
    err_est_combined = integ.opts.internalnorm(err_tmp, t)  # Norm over the dimensions
    integ.EEst = err_est_combined

    integ.cache.x = PI * integ.cache.x
    integ.cache.x_pred = PI * integ.cache.x_pred
    integ.cache.x_filt = PI * integ.cache.x_filt
end


function predict!(integ::ODEFilterIntegrator)

    @unpack dt = integ
    @unpack A!, Q!, InvPrecond = integ.cache
    @unpack x, Ah, Qh, x_pred = integ.cache

    A!(Ah, dt)
    Q!(Qh, dt)

    pred = predict(x, Ah, Qh)
    copy!(x_pred, pred)

    return x_pred
end


function measure_h!(integ::ODEFilterIntegrator, x_pred, t)

    @unpack p, f, dt = integ
    @unpack E0, h!, InvPrecond = integ.cache
    @unpack du, h, u_pred = integ.cache
    PI = InvPrecond(dt)

    IIP = isinplace(integ)
    if IIP
        f(du, u_pred, p, t)
    else
        du .= f(u_pred, p, t)
    end
    integ.destats.nf += 1

    h!(h, du, PI*x_pred.μ)
end

function measure_H!(integ::ODEFilterIntegrator, x_pred, t)

    @unpack p, f, dt = integ
    @unpack jac, H!, InvPrecond = integ.cache
    @unpack u_pred, ddu, H = integ.cache
    PI = InvPrecond(dt)

    if !isnothing(jac)
        if isinplace(integ)
            jac(ddu, u_pred, p, t)
        else
            ddu .= jac(u_pred, p, t)
        end
        integ.destats.njacs += 1
    end
    H!(H, ddu)
    H .= H * PI
end

function measure!(integ, x_pred, t)
    measure_h!(integ, x_pred, t)
    measure_H!(integ, x_pred, t)

    @unpack dt = integ
    @unpack R, q, d = integ.cache
    @unpack measurement, h, H = integ.cache

    v, S = measurement.μ, measurement.Σ
    v .= 0 .- h
    R .= Diagonal(eps.(v))
    S .= Symmetric(H * x_pred.Σ * H' .+ R)

    return nothing
end


function update!(integ::ODEFilterIntegrator, prediction)

    @unpack dt = integ
    @unpack R, q, d, Precond, InvPrecond, E1 = integ.cache
    @unpack measurement, h, H, K, x_filt = integ.cache
    P, PI = Precond(dt), InvPrecond(dt)

    v, S = measurement.μ, measurement.Σ

    m_p, P_p = prediction.μ, prediction.Σ

    S_inv = inv(S)
    K .= P_p * H' * S_inv

    x_filt.μ .= m_p .+ K*v

    # Joseph Form
    x_filt.Σ .= Symmetric(X_A_Xt(PDMat(Symmetric(P_p)), (I-K*H)))
    if !iszero(R)
        x_filt.Σ .+= Symmetric(X_A_Xt(PDMat(R), K))
    end

    assert_nonnegative_diagonal(x_filt.Σ)

    return x_filt
end
