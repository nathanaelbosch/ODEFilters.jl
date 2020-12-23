"""
EK0 v2.0, aka. FastEK0!

This file contains everything that relates to it, including notably
- The algorithm: `FastEK0 <: AbtractEK`
- 2 types of caches: `FastEK0ConstantCache` and `FastEK0Cache`, for OOP and IIP;
  `FastEK0Cache` actually calls `FastEK0ConstantCache`! They might also get
  partially merged at some point, now that the implementation seems more fixed.
- 2 implementations of `perform_step!` (IIP and OOP)
- an `initialize!` for both versions
- a custom `postamble!` since some things need to be overwritten, though this
  might be removed later
- soon: A custom `smooth!` implementation
"""

Base.@kwdef struct FastEK0 <: AbstractEK
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end


mutable struct FastEK0ConstantCache{
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
    alg::FastEK0, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP::Val{false})

    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    @assert alg.diffusionmodel == :dynamic "FastEK0 only uses dynamic diffusion"
    @assert u isa AbstractVector "Only vector-valued problems are supported"

    q = alg.order
    u0 = u
    t0 = t
    d = length(u)

    # Proj is never called right now anyways!
    Proj(deriv) = KronMat(reshape([i==(deriv+1) ? 1 : 0 for i in 1:q+1], (1, q+1)), d)
    SolProj = Proj(0)
    I0, I1 = 1:d, d+1:2d  # Slicing instead of projection matrices

    Precond = invpreconditioner(1, q)
    _, Q = ibm(1, q, uEltypeNoUnits)
    A, _ = vanilla_ibm(1, q, uEltypeNoUnits)
    R = zeros(d, d)

    m0, P0 = initialize_with_derivatives(Vector(u0), f, p, t0, q)
    @assert iszero(P0)
    # Exploit the EK0's Kronecker structure
    x0 = Gaussian(m0, SquarerootMatrix(KronMat(P0[1:d:d*(q+1), 1:d:d*(q+1)], d)))
    # Predictions always use adjoints of upper triangular matrices
    xpred = Gaussian(copy(m0), SquarerootMatrix(KronMat(
        UpperTriangular(P0[1:d:d*(q+1), 1:d:d*(q+1)])', d)))

    initdiff = one(uEltypeNoUnits)  # "dynamic diffusion" is a hard choice here
    ll = zero(uEltypeNoUnits)

    return FastEK0ConstantCache(
        d, q, A, Q, R, Proj, SolProj, Precond, x0, initdiff, I0, I1, ll, xpred)
end


function OrdinaryDiffEq.perform_step!(integ, cache::FastEK0ConstantCache, repeat_step=false)
    @unpack t, dt, f, p = integ
    @unpack d, q, Proj, SolProj, Precond, I0, I1 = integ.cache
    @unpack x, A, Q, R = integ.cache

    tnew = t + dt

    # Setup
    Tinv = Precond(dt)
    Ah = A(dt)
    HQH = Tinv[2,2]^2 * Q[2,2]

    m, P = x.μ, x.Σ
    PL = P.squareroot.left

    # Predict - Mean
    mp = vec(reshape(m, (d, q+1)) * Ah')  # m_p = Ah*m

    u_pred = @view mp[I0]  # E0 * m_p
    du_pred = @view mp[I1]  # E1 * m_p

    # Measure
    du = f(u_pred, p, tnew)
    integ.destats.nf += 1
    z_neg = du - du_pred  # having the negative saves an allocation later

    # Calibrate
    # σ² = z' * (HQH \ z) / d  # z'inv(H*Q*H')z / d
    σ² = z_neg'z_neg / HQH  # = z' * (HQH \ z) / d
    cache.diffusion = σ²

    # Predict - Cov
    QhL_calibrated = sqrt(σ²) .* Tinv.diag .* Q.L
    small_qr_input = [Ah*PL QhL_calibrated]'
    Pp = small_qr_input'small_qr_input
    chol = cholesky(Pp, check=false)
    PpL = issuccess(chol) ? chol.U' : qr(small_qr_input).R'  # only use QR if necessary
    x_pred = Gaussian(mp, SquarerootMatrix(KronMat(PpL, d)))
    integ.cache.x_pred = x_pred

    # Measurement Cov
    @assert iszero(R)
    S = Pp[2,2]  # = H*Pp*H'

    # Update
    K = Pp[:, 2] / S  # = Pp * H' * inv(S)
    mf = mp .+ vec((z_neg)*K')  # = mp + K*(0-z)
    PfL = PpL - K*(@view PpL[2, :])'  # = Pp - K*S*K'
    Pf = SquarerootMatrix(KronMat(PfL, d))

    x_filt = Gaussian(mf, Pf)
    integ.u = mf[I0]

    # Estimate error for adaptive steps
    if integ.opts.adaptive

        err_est_unscaled = sqrt(σ²*HQH)  # = .√diag(σ²*HQH')

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


#################################################################################
# IIP definition with pre-allocation and everything!
mutable struct FastEK0Cache{
    AType, QType, RType, ProjType, SolProjType, FP, xType, diffusionType,
    llType, IType, xpType,
    # Mutable stuff
    uType, mType, PType, KType, MType, PreQRType, QLType
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
    # Indices for projections
    I0::IType
    I1::IType
    # Nice to have
    log_likelihood::llType
    x_pred::xpType

    # Stuff to pre-allocate
    u::uType
    tmp::uType
    m_tmp::mType
    P_tmp::PType
    P_tmp2::PType
    P_tmp3::PType
    K_tmp::KType
    M_tmp::MType
    preQRmat::PreQRType
    QL_tmp::QLType
end
function OrdinaryDiffEq.alg_cache(
    alg::FastEK0, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP::Val{true})

    constants = OrdinaryDiffEq.alg_cache(alg, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, Val(false))

    @unpack d, q, x = constants

    D = d*(q+1)
    m_tmp = zeros(uEltypeNoUnits, D)

    D = q+1  # We live in kronecker world!

    # Create some arrays to cache into
    P_tmp = zeros(uEltypeNoUnits, D, D)
    P_tmp2 = zeros(uEltypeNoUnits, D, D)
    P_tmp3 = zeros(uEltypeNoUnits, D, D)
    K_tmp = zeros(uEltypeNoUnits, D)
    M_tmp = zeros(uEltypeNoUnits, d, q+1)  # For reshaped means
    preQRmat = zeros(uEltypeNoUnits, D, 2D)

    _, QL = ibm(1, q, uEltypeNoUnits)
    QL_tmp = QL.L
    A, Q = vanilla_ibm(1, q, uEltypeNoUnits)


    return FastEK0Cache(
        constants.d, constants.q, constants.A, Q, constants.R, constants.Proj,
        constants.SolProj, constants.Precond, constants.x, constants.diffusion,
        constants.I0, constants.I1, constants.log_likelihood, constants.x_pred,
        # Mutable stuff
        copy(u), copy(u),
        m_tmp, P_tmp, P_tmp2, P_tmp3,
        K_tmp, M_tmp, preQRmat, QL_tmp,
    )
end


function OrdinaryDiffEq.perform_step!(integ, cache::FastEK0Cache, repeat_step=false)
    @unpack t, dt, f, p = integ
    @unpack d, q, Proj, SolProj, Precond, I0, I1 = integ.cache
    @unpack x, A, Q, R = integ.cache

    # Load pre-allocated stuff, and assign them to more meaningful variables
    @unpack tmp, m_tmp, P_tmp, P_tmp2, K_tmp, M_tmp, preQRmat = integ.cache
    @unpack QL_tmp = integ.cache
    QL = QL_tmp

    tnew = t + dt

    # Setup
    Ah = A(dt)
    Qh = Q(dt)
    HQH = Qh[2,2]

    m, P = x.μ, x.Σ
    PL = P.squareroot.left

    # Predict - Mean
    # TODO This is comparatively slow and allocates! The first should be best!
    m_p = copyto!(m_tmp, mul!(M_tmp, reshape(m, (d, q+1)), Ah'))
    m_p = vec(mul!(M_tmp, reshape(m, (d, q+1)), Ah'))
    m_p = vec(reshape(m, (d, q+1)) * Ah')

    u_pred = @view m_p[I0]  # E0 * m_p
    du_pred = @view m_p[I1]  # E1 * m_p

    # Measure
    du = cache.u
    f(du, u_pred, p, tnew)
    integ.destats.nf += 1
    du .-= du_pred
    z_neg = du

    # Calibrate
    σ² = z_neg'z_neg / HQH
    cache.diffusion = σ²

    # Predict - Cov
    mul!(P_tmp2, Ah, PL)
    mul!(P_tmp, P_tmp2, P_tmp2')
    @. P_tmp += σ²*Qh
    Pp = P_tmp
    chol = cholesky(Symmetric(P_tmp), check=false)
    PpL = chol.U'
    if !issuccess(chol)
        preQRmat[:, 1:q+1] .= P_tmp2
        TI = Precond(dt)
        @. preQRmat[:, q+2:end] = sqrt(σ²) * TI.diag * QL
        PpL = qr(preQRmat').R'
    end
    copy!(integ.cache.x_pred.μ, m_p)
    copy!(integ.cache.x_pred.Σ.squareroot.left, PpL)

    # Measurement Cov
    @assert iszero(R)
    S = Pp[2,2]

    # Update
    K_neg = K_tmp
    @. K_neg = (@view Pp[:, 2]) / -S  # P_p * H' * inv(S)
    # The following computes m_tmp such that: @assert m_tmp ≈ kron(K_neg, z_neg)
    for i in 1:q+1 @. m_tmp[(i-1)*d+1:i*d] = K_neg[i]*z_neg end
    m_f = m_p .-= m_tmp
    PfL = P_tmp  # => Pp will be overwritten now
    mul!(PfL, K_neg, PpL[2, :]')  # P_f = P_p - K*S*K'
    PfL .+= PpL
    P_f = SquarerootMatrix(KronMat(PfL, d))

    x_filt = Gaussian(m_f, P_f)
    integ.u .= @view m_f[I0]

    # Estimate error for adaptive steps
    if integ.opts.adaptive

        err_tmp = sqrt(σ²*HQH)

        DiffEqBase.calculate_residuals!(
            tmp, dt * err_tmp, integ.u, integ.uprev,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(tmp, t) # scalar

    end
    if !integ.opts.adaptive || integ.EEst < one(integ.EEst)
        copy!(integ.cache.x, x_filt)
        integ.sol.log_likelihood += integ.cache.log_likelihood
    end
end


function OrdinaryDiffEq.initialize!(integ, cache::Union{FastEK0ConstantCache, FastEK0Cache})
    # @assert integ.opts.dense == integ.alg.smooth "`dense` and `smooth` should have the same value! "
    @assert integ.saveiter == 1
    if integ.opts.dense
        OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, cache.x)
    end
    # OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj*cache.x)
    d = integ.cache.d
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, Gaussian(integ.u, integ.cache.x.Σ.squareroot.left[1,1]^2*I(d)))
end




function OrdinaryDiffEq.postamble!(integ::OrdinaryDiffEq.ODEIntegrator{<:FastEK0})
    @debug "postamble!"

    if integ.alg.smooth
        smooth_all!(integ, integ.cache)
        d = integ.cache.d
        for i in 1:length(integ.sol.pu)
            integ.sol.pu[i] = Gaussian(integ.sol.x[i].μ[1:d], integ.sol.x[i].Σ.squareroot.left[1,1]^2*I(d))
        end
        @assert (length(integ.sol.u) == length(integ.sol.pu))
        [(su .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
    end

    OrdinaryDiffEq._postamble!(integ)
end

function smooth_all!(integ, cache::Union{FastEK0ConstantCache, FastEK0Cache})
    @unpack x, x_preds, t, diffusions = integ.sol
    @unpack A, Q, Precond, d, q = integ.cache

    for i in length(x)-1:-1:2
        dt = t[i+1] - t[i]
        # The "current" state that we want to smooth: x[i]
        # The "next" state, that is assumed to be smoothed: x[i+1]
        # The estimated diffusion for this interval diffusions[i]

        Ah = A(dt)
        m, P = x[i].μ, x[i].Σ
        PL = P.squareroot.left
        ms, Ps = x[i+1].μ, x[i+1].Σ
        PsL = Ps.squareroot.left
        mp, Pp = x_preds[i].μ, x_preds[i].Σ  # x_preds is 1 shorter
        PpL = Pp.squareroot.left

        mnew, PnewL = smooth_step(m, PL, mp, PpL, ms, PsL, Ah, d, q)

        Pnew = SquarerootMatrix(KronMat(PnewL, d))
        x[i] = Gaussian(mnew, Pnew)
    end
end

function smooth_step(m, PL, mp, PpL, ms, PsL, Ah, d, q, T=nothing)
    PpLinv = inv(PpL)
    Ppinv = PpLinv'PpLinv
    G = (PL*PL') * Ah' * Ppinv
    # mnew = m + vec(reshape((ms - mp), (d, q+1)) * G')
    if length(m) != size(G)[1]
        mout = m + vec(reshape((ms - mp), (d, q+1)) * G')
    else
        mout = m + G*(ms - mp)
    end

    # Would this work?
    # small_qr_input = [PL G*(PpL-PsL)]'
    Pout = (PL*PL') + G * (PsL*PsL' - PpL*PpL') * G'
    @info "._." G (PL*PL') Pout (PsL*PsL' - PpL*PpL')
    PLout = cholesky(Symmetric(Pout)).L

    # TI = inv(T)
    # @info "smooth step" TI*m TI*mout TI*G*T
    # if any(abs.(m) .> 1e4)
    #     @info "Something went wrong during smoothing" m mout G
    #     error("Something's Fishy")
    # end

    return mout, PLout
end
