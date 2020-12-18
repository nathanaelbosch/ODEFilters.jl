Base.@kwdef struct FastEK0 <: AbstractEK
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end


mutable struct FastEK0ConstantCache{
    AType, QType,
    RType,
    ProjType,
    SolProjType,
    FP,
    xType,
    diffusionType,
    llType,
    IType,
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
end


function OrdinaryDiffEq.alg_cache(
    alg::FastEK0, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP::Val{false})

    q = alg.order
    u0 = u
    t0 = t
    d = length(u)

    Proj(deriv) = KronMat(reshape([i==(deriv+1) ? 1 : 0 for i in 1:q+1], (1, q+1)), d)
    # Proj(deriv) = kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    SolProj = Proj(0)

    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    @assert alg.diffusionmodel == :dynamic "FastEK0 only uses dynamic diffusion"

    Precond = invpreconditioner(1, q)
    _, Q = ibm(1, q, uEltypeNoUnits)
    A, _ = vanilla_ibm(1, q, uEltypeNoUnits)
    R = zeros(d, d)

    m0, P0 = initialize_with_derivatives(Vector(u0), f, p, t0, q)
    if u0 isa StaticArray
        m0 = SVector{length(m0)}(m0)
        P0 = SMatrix{size(P0)...}(P0)
        # A = SMatrix{size(A)...}(A)
        # Q = SMatrix{size(Q)...}(Q)
        R = SMatrix{size(R)...}(R)
    end
    # P0 = PSDMatrix(LowerTriangular(P0))
    @assert iszero(P0)
    KI = 1:d:d*(q+1)
    P0 = SquarerootMatrix(KronMat(P0[KI, KI], d))
    # P0 = SquarerootMatrix(P0)
    x0 = Gaussian(m0, P0)

    initdiff = one(uEltypeNoUnits)

    # I0, I1 = SVector{d}(1:d), SVector{d}(d+1:2d)
    I0, I1 = 1:d, d+1:2d

    return FastEK0ConstantCache{
        typeof(A), typeof(Q),
        typeof(R), typeof(Proj), typeof(SolProj), typeof(Precond),
        typeof(x0), uEltypeNoUnits, uEltypeNoUnits,
        typeof(I0)
    }(
        d, q, A, Q, R, Proj, SolProj, Precond,
        x0, initdiff,
        I0, I1,
        zero(uEltypeNoUnits),
    )
end


function OrdinaryDiffEq.perform_step!(integ, cache::FastEK0ConstantCache, repeat_step=false)
    @unpack t, dt, f, p = integ
    @unpack d, q, Proj, SolProj, Precond, I0, I1 = integ.cache
    @unpack x, A, Q, R = integ.cache
    D = d*(q+1)

    tnew = t + dt

    # Setup
    TI = Precond(dt)
    Ah = A(dt)
    QhL = TI*Q.L
    KI = 1:d:D
    smallAh = Ah
    smallQhL = QhL
    HQhL = @view QhL[2, :]
    HQH = HQhL'HQhL

    m, P = x.μ, x.Σ
    # PL = @view P.squareroot[KI, KI]
    PL = P.squareroot.left

    # Predict - Mean
    # m_p = Ah*m
    m_p = vec(reshape(m, (d, q+1)) * smallAh')
    # @assert m_p == m_p_tricked
    u_pred = @view m_p[I0]  # E0 * m_p
    du_pred = @view m_p[I1]  # E1 * m_p

    # Measure
    du = f(u_pred, p, tnew)
    z_neg = du - du_pred

    # Calibrate
    # σ² = z' * (HQH \ z) / d  # z'inv(H*Q*H')z / d
    σ² = z_neg'z_neg / HQH
    cache.diffusion = σ²

    # Predict - Cov
    QhL_calibrated = sqrt(σ²) * QhL
    small_qr_input = [Ah*PL QhL_calibrated]'
    Pp = small_qr_input'small_qr_input
    PpL = cholesky(Pp).L
    # If this fails, replace with
    # PpL = qr(small_qr_input).R'

    # Measurement Cov
    # @assert iszero(R)
    SL = @view PpL[2, :]
    S = SL'SL

    # Update
    Sinv = inv(S)
    # K = PpL * PpL[2, :] * Sinv  # P_p * H' * inv(S)
    K = Pp[:, 2] * Sinv  # P_p * H' * inv(S)
    m_f = m_p .+ vec((z_neg)*K')
    PfL = PpL - K*(@view PpL[2, :])'  # P_f = P_p - K*S*K'
    # P_f = SquarerootMatrix(kron(PfL, I(d)))
    P_f = SquarerootMatrix(KronMat(PfL, d))

    x_filt = Gaussian(m_f, P_f)
    integ.u = m_f[I0]

    # Estimate error for adaptive steps
    if integ.opts.adaptive

        err_est_unscaled = sqrt(σ²*HQH)

        err = DiffEqBase.calculate_residuals(
            dt * err_est_unscaled, integ.u, integ.uprev,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(err, t) # scalar

        if integ.EEst < one(integ.EEst)
            integ.cache.x = x_filt
            integ.sol.log_likelihood += integ.cache.log_likelihood
        end
    end
end


#################################################################################
# IIP definition with pre-allocation and everything!
mutable struct FastEK0Cache{
    AType, QType,
    RType,
    ProjType,
    SolProjType,
    FP,
    xType,
    diffusionType,
    llType,
    IType,
    # Mutable stuff
    uType,
    mType, PType, SType, KType, PreQRType,
    QLType
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
    # Stuff to pre-allocate
    u::uType
    # u_pred::uType
    # u_filt::uType
    tmp::uType
    # x_pred::xType
    # x_filt::xType
    # x_tmp::xType
    # x_tmp2::xType
    # measurement::measType
    # H::matType
    # du::uType
    # ddu::matType
    # K::matType
    # G::matType
    # covmatcache::matType
    # err_tmp::uType

    x_filt::xType
    m_tmp::mType
    m_tmp2::mType
    P_tmp::PType
    P_tmp2::PType
    u_tmp::uType
    u_tmp2::uType
    z_tmp::uType
    S_tmp::SType
    S_tmp2::SType
    K_tmp::KType
    K_tmp2::KType
    preQRmat::PreQRType
    QL_tmp::QLType
end
function OrdinaryDiffEq.alg_cache(
    alg::FastEK0, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP::Val{true})

    constants = OrdinaryDiffEq.alg_cache(alg, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, Val(false))

    @unpack d, q, x = constants

    D = d*(q+1)
    m_tmp = zeros(uEltypeNoUnits, D)
    m_tmp2 = zeros(uEltypeNoUnits, D)

    D = q+1

    # Create some arrays to cache into
    d = 1
    cov = zeros(uEltypeNoUnits, D, D)
    K = zeros(uEltypeNoUnits, D, D)
    pre_QR = zeros(uEltypeNoUnits, D, 2D)
    pre_QR = zeros(uEltypeNoUnits, D, 2D)

    P_tmp = zeros(uEltypeNoUnits, D, D)
    P_tmp2 = zeros(uEltypeNoUnits, D, D)
    u_tmp = copy(u)
    u_tmp2 = copy(u)
    z_tmp = copy(u)
    S_tmp = zeros(uEltypeNoUnits, d, d)
    S_tmp2 = zeros(uEltypeNoUnits, d, d)
    K_tmp = zeros(uEltypeNoUnits, D, d)
    K_tmp2 = zeros(uEltypeNoUnits, D, d)
    preQRmat = zeros(uEltypeNoUnits, D, 2D)
    QL_tmp = LowerTriangular(zeros(uEltypeNoUnits, D, D))


    return FastEK0Cache{
        typeof(constants.A), typeof(constants.Q),
        typeof(constants.R), typeof(constants.Proj), typeof(constants.SolProj),
        typeof(constants.Precond),
        typeof(constants.x), typeof(constants.diffusion), typeof(constants.log_likelihood),
        typeof(constants.I0),
        # Mutable stuff
        typeof(u),
        typeof(m_tmp), typeof(P_tmp), typeof(S_tmp), typeof(K_tmp), typeof(preQRmat),
        typeof(QL_tmp),
    }(
        constants.d, constants.q, constants.A, constants.Q, constants.R, constants.Proj, constants.SolProj, constants.Precond,
        constants.x, constants.diffusion,
        constants.I0, constants.I1,
        constants.log_likelihood,
        # Mutable stuff
        copy(u), copy(u),
        copy(constants.x),
        m_tmp, m_tmp2, P_tmp, P_tmp2, u_tmp, u_tmp2, z_tmp, S_tmp, S_tmp2,
        K_tmp, K_tmp2, preQRmat,
        QL_tmp,
    )
end


function OrdinaryDiffEq.perform_step!(integ, cache::FastEK0Cache, repeat_step=false)
    @unpack t, dt, f, p = integ
    @unpack d, q, Proj, SolProj, Precond, I0, I1 = integ.cache
    @unpack x, A, Q, R = integ.cache

    D = d*(q+1)

    # Load pre-allocated stuff, and assign them to more meaningful variables
    @unpack tmp, m_tmp, m_tmp2, P_tmp, P_tmp2, z_tmp, u_tmp, u_tmp2, S_tmp, S_tmp2, K_tmp, K_tmp2, preQRmat = integ.cache
    @unpack QL_tmp = integ.cache
    @unpack x_filt = integ.cache
    err_tmp = u_tmp
    HQH = S_tmp2
    cov_tmp = P_tmp

    tnew = t + dt

    # Setup
    TI = Precond(dt)
    Ah = A(dt)
    QhL = QL_tmp
    @. QhL = TI.diag * Q.L
    HQhL = @view QhL[2, :]
    HQH = HQhL'HQhL

    m, P = x.μ, x.Σ
    KI = 1:d:D
    # PL = @view P.squareroot[KI, KI]
    PL = P.squareroot.left

    # Predict - Mean
    m_p = vec(reshape(m, (d, q+1)) * Ah')
    u_pred = @view m_p[I0]  # E0 * m_p
    du_pred = @view m_p[I1]  # E1 * m_p

    # Measure
    du = cache.u
    f(du, u_pred, p, tnew)
    du .-= du_pred
    z_neg = du

    # Calibrate
    # σ² = z' * (HQH \ z) / d  # z'inv(H*Q*H')z / d
    σ² = z_neg'z_neg / HQH
    cache.diffusion = σ²

    # Predict - Cov
    preQRmat[:, 1:q+1] .= Ah*PL
    @. preQRmat[:, q+2:end] = sqrt(σ²) * QhL
    mul!(P_tmp2, preQRmat, preQRmat')
    Pp = P_tmp2
    copy!(P_tmp, Pp)
    PpL = cholesky!(P_tmp).L
    # If this fails, replace with
    # PpL = qr(preQRmat').R'

    # Measurement Cov
    # @assert iszero(R)
    SL = @view PpL[2, :]
    S = SL'SL

    # Update
    Sinv = inv(S)
    # K = PpL * PpL[2, :] * Sinv  # P_p * H' * inv(S)
    K = K_tmp
    @. K = (@view Pp[:, 2]) * Sinv  # P_p * H' * inv(S)
    # P_p * H' * inv(S)
    m_p .+= vec(z_neg*K')
    m_f = m_p
    PfL = P_tmp
    PfL .= PpL .- K*(@view PpL[2, :])'  # P_f = P_p - K*S*K'
    P_f = SquarerootMatrix(KronMat(PfL, d))
    # P_f = SquarerootMatrix(kron(PfL, I(d)))
    x_filt = Gaussian(m_f, P_f)
    integ.u .= @view m_f[I0]

    # Estimate error for adaptive steps
    if integ.opts.adaptive

        err_tmp = sqrt(σ²*HQH)

        DiffEqBase.calculate_residuals!(
            tmp, dt * err_tmp, integ.u, integ.uprev,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(tmp, t) # scalar

        if integ.EEst < one(integ.EEst)
            copy!(integ.cache.x, x_filt)
            integ.sol.log_likelihood += integ.cache.log_likelihood
        end
    end
end


function OrdinaryDiffEq.initialize!(integ, cache::Union{FastEK0ConstantCache, FastEK0Cache})
    @assert integ.saveiter == 1
    if integ.opts.dense
        OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, cache.x)
    end
    # OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj*cache.x)
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, Gaussian(integ.u, integ.cache.x.Σ.squareroot.left[1,1]^2*I))
end
