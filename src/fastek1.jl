Base.@kwdef struct FastEK1 <: AbstractEK
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end


mutable struct FastEK1ConstantCache{
    AType, QType, RType, ProjType, SolProjType, FP, xType, diffusionType,
    llType, IType, xpType
} <: ODEFiltersCache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    A::AType
    Q::QType
    # diffusionmodel::diffModelType
    R::RType
    Proj::ProjType
    SolProj::SolProjType
    Precond::FP
    # NEEDS to be tracked
    x::xType
    diffusion::diffusionType
    # Indices for projections; Faster than using `Proj`!
    I0::IType
    I1::IType
    # Nice to have
    log_likelihood::llType
    x_pred::xpType
end


function OrdinaryDiffEq.alg_cache(
    alg::FastEK1, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP::Val{false})

    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    @assert alg.diffusionmodel == :dynamic "FastEK1 only uses dynamic diffusion"
    @assert u isa AbstractVector "Only vector-valued problems are supported"

    q = alg.order
    u0 = u
    t0 = t
    d = length(u)

    # Proj is never called right now anyways!
    Proj(deriv) = kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', I(d))
    SolProj = Proj(0)
    I0, I1 = 1:d, d+1:2d  # Slicing instead of projection matrices

    Precond = preconditioner(d, q)
    _, Q = ibm(d, q, uEltypeNoUnits)
    A, _ = vanilla_ibm(d, q, uEltypeNoUnits)
    R = zeros(d, d)

    m0, P0 = initialize_with_derivatives(Vector(u0), f, p, t0, q)
    @assert iszero(P0)
    # Exploit the EK1's Kronecker structure
    x0 = Gaussian(m0, SquarerootMatrix(P0))
    # Predictions always use adjoints of upper triangular matrices
    xpred = Gaussian(copy(m0), SquarerootMatrix(UpperTriangular(P0)'))

    initdiff = one(uEltypeNoUnits)  # "dynamic diffusion" is a hard choice here
    ll = zero(uEltypeNoUnits)

    return FastEK1ConstantCache(
        d, q, A, Q, R, Proj, SolProj, Precond, x0, initdiff, I0, I1, ll, xpred)
end


function OrdinaryDiffEq.perform_step!(integ, cache::FastEK1ConstantCache, repeat_step=false)
    @unpack t, dt, f, p = integ
    @unpack d, q, Proj, SolProj, Precond, I0, I1 = integ.cache
    @unpack x, A, Q, R = integ.cache

    tnew = t + dt

    # Setup
    Tinv = inv(Precond(dt))
    Ah = A(dt)
    Qh = Tinv*Q*Tinv
    # HQH = Tinv[2,2]^2 * Q[2,2]

    m, P = x.μ, x.Σ
    PL = P.squareroot

    # Predict - Mean
    mp = Ah*m

    u_pred = @view mp[I0]  # E0 * m_p
    du_pred = @view mp[I1]  # E1 * m_p

    # Measure
    du = f(u_pred, p, tnew)
    integ.destats.nf += 1
    z_neg = du - du_pred  # having the negative saves an allocation later

    # Measure Jac and build H
    ddu = f.jac(u_pred, p, t)
    integ.destats.njacs += 1
    E0, E1 = Proj(0), Proj(1)
    H = (E1 - ddu*E0)
    HQH = H*Q*H'

    # Calibrate
    σ² = z_neg' * (HQH \ z_neg) / d  # z'inv(H*Q*H')z / d
    cache.diffusion = σ²

    # Predict - Cov
    QhL_calibrated = sqrt(σ²) .* Tinv.diag .* Q.L
    small_qr_input = [Ah*PL QhL_calibrated]'
    Pp = small_qr_input'small_qr_input
    chol = cholesky(Pp, check=false)
    PpL = issuccess(chol) ? chol.U' : qr(small_qr_input).R'  # only use QR if necessary
    x_pred = Gaussian(mp, SquarerootMatrix(PpL))
    integ.cache.x_pred = x_pred

    # Measurement Cov
    @assert iszero(R)
    S = H*Pp*H'

    # Update
    Sinv = inv(S)
    K = Pp * H' * Sinv
    mf = mp .+ K*z_neg
    PfL = PpL - K*H*PpL
    Pf = SquarerootMatrix(PfL)

    x_filt = Gaussian(mf, Pf)
    integ.u = mf[I0]

    # Estimate error for adaptive steps
    if integ.opts.adaptive

        err_est_unscaled = sqrt.(σ².*diag(HQH))  # = .√diag(σ²*HQH')

        err = DiffEqBase.calculate_residuals(
            dt * err_est_unscaled, integ.u, integ.uprev,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(err, t)
    end
    if !integ.opts.adaptive || integ.EEst < one(integ.EEst)
        integ.cache.x = x_filt
        integ.sol.log_likelihood += integ.cache.log_likelihood

    end
end


function OrdinaryDiffEq.initialize!(integ, cache::Union{FastEK1ConstantCache})
    # @assert integ.opts.dense == integ.alg.smooth "`dense` and `smooth` should have the same value! "
    @assert integ.saveiter == 1
    if integ.opts.dense
        OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, cache.x)
    end
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj*cache.x)
end



function OrdinaryDiffEq.postamble!(integ::OrdinaryDiffEq.ODEIntegrator{<:FastEK1})
    @debug "postamble!"

    if integ.alg.smooth
        smooth_all!(integ, integ.cache)
        d = integ.cache.d
        for i in 1:length(integ.sol.pu)
            integ.sol.pu[i] = integ.cache.SolProj*integ.sol.x[i]
        end
        @assert (length(integ.sol.u) == length(integ.sol.pu))
        [(su .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
    end

    OrdinaryDiffEq._postamble!(integ)
end

function smooth_all!(integ, cache::FastEK1ConstantCache)
    @unpack x, x_preds, t, diffusions = integ.sol
    @unpack A, Q, Precond, d, q = integ.cache

    for i in length(x)-1:-1:2
        dt = t[i+1] - t[i]
        # The "current" state that we want to smooth: x[i]
        # The "next" state, that is assumed to be smoothed: x[i+1]
        # The estimated diffusion for this interval diffusions[i]

        Ah = A(dt)
        # Tinv = Precond(dt)
        m, P = x[i].μ, x[i].Σ

        PL = P.squareroot
        ms, Ps = x[i+1].μ, x[i+1].Σ
        PsL = Ps.squareroot
        mp, Pp = x_preds[i].μ, x_preds[i].Σ  # x_preds is 1 shorter
        PpL = Pp.squareroot
        σ² = diffusions[i]

        PpLinv = inv(PpL)
        Ppinv = PpLinv'PpLinv
        G = PL*(PL' * (Ah' * Ppinv))
        # mnew = m + vec(reshape((ms - mp), (d, q+1)) * G')
        m .+= G * (ms .- mp)

        # Joseph-Form:
        # PnewL = PL - G*(PsL-PpL)
        PL .-= G*(PsL.-PpL)
        mnew, PnewL = m, PL
        Pnew = SquarerootMatrix(PnewL)
        x[i] = Gaussian(mnew, Pnew)
    end
end
