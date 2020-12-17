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

    Proj(deriv) = kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    SolProj = Proj(0)

    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    @assert alg.diffusionmodel == :dynamic "FastEK0 only uses dynamic diffusion"

    Precond = preconditioner(d, q)
    A, Q = ibm(d, q, uEltypeNoUnits)
    # A, Q = vanilla_ibm(d, q, uEltypeNoUnits)
    # Q = Matrix(Q)
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
    P0 = SquarerootMatrix(P0)
    x0 = Gaussian(m0, P0)

    initdiff = one(uEltypeNoUnits)

    I0, I1 = SVector{d}(1:d), SVector{d}(d+1:2d)

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
    T = Precond(dt); TI = inv(T)
    Ah, QhL = TI*A*T, TI*Q.L
    HQhL = @view QhL[I1, :]
    HQH = HQhL*HQhL'

    m, P = x.μ, x.Σ
    PL = P.squareroot

    # Predict - Mean
    m_p = Ah*m
    u_pred = @view m_p[I0]  # E0 * m_p
    du_pred = @view m_p[I1]  # E1 * m_p

    # Measure
    du = f(u_pred, p, tnew)
    z = du_pred - du

    # Calibrate
    σ² = z' * (HQH \ z) / d  # z'inv(H*Q*H')z / d
    cache.diffusion = σ²

    # Predict - Cov
    QhL_calibrated = sqrt(σ²) * QhL
    _, PpR = qr([Ah*PL QhL_calibrated]')
    PpL = PpR'

    # Measurement Cov
    @assert iszero(R)
    SL = PpL[I1, :]
    S = SL*SL'

    # Update
    Sinv = inv(S)
    K = PpL * (PpL[I1, :])' * Sinv  # P_p * H' * inv(S)
    m_f = m_p .+ K * (0 .- z)
    PfL = PpL - K*PpL[I1, :]  # P_f = P_p - K*S*K'
    P_f = SquarerootMatrix(PfL)

    x_filt = Gaussian(m_f, P_f)
    integ.u = m_f[I0]

    # Estimate error for adaptive steps
    if integ.opts.adaptive

        err_est_unscaled = sqrt.(σ².*diag(HQH))

        err = DiffEqBase.calculate_residuals(
            dt * err_est_unscaled, integ.u, integ.uprev,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(err, t) # scalar

        if integ.EEst < one(integ.EEst)
            integ.cache.x = x_filt
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
end
function OrdinaryDiffEq.alg_cache(
    alg::FastEK0, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP::Val{true})
    constants = OrdinaryDiffEq.alg_cache(alg, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, Val(false))

    return FastEK0Cache{
        typeof(constants.A), typeof(constants.Q),
        typeof(constants.R), typeof(constants.Proj), typeof(constants.SolProj),
        typeof(constants.Precond),
        typeof(constants.x), typeof(constants.diffusion), typeof(constants.log_likelihood),
        typeof(constants.I0),
        # Mutable stuff
        typeof(u),
    }(
        constants.d, constants.q, constants.A, constants.Q, constants.R, constants.Proj, constants.SolProj, constants.Precond,
        constants.x, constants.diffusion,
        constants.I0, constants.I1,
        constants.log_likelihood,
        # Mutable stuff
        copy(u), copy(u),
    )
end

function OrdinaryDiffEq.perform_step!(integ, cache::FastEK0Cache, repeat_step=false)
    @unpack t, dt, f, p = integ
    @unpack d, q, Proj, SolProj, Precond, I0, I1 = integ.cache
    @unpack x, A, Q, R = integ.cache
    D = d*(q+1)

    tnew = t + dt

    # Setup
    T = Precond(dt); TI = inv(T)
    Ah, QhL = TI*A*T, TI*Q.L
    HQhL = @view QhL[I1, :]
    HQH = HQhL*HQhL'

    m, P = x.μ, x.Σ
    PL = P.squareroot

    # Predict - Mean
    m_p = Ah*m
    u_pred = @view m_p[I0]  # E0 * m_p
    du_pred = @view m_p[I1]  # E1 * m_p

    # Measure
    du = cache.u
    f(du, u_pred, p, tnew)
    z = du_pred - du

    # Calibrate
    σ² = z' * (HQH \ z) / d  # z'inv(H*Q*H')z / d
    cache.diffusion = σ²

    # Predict - Cov
    QhL_calibrated = sqrt(σ²) * QhL
    _, PpR = qr([Ah*PL QhL_calibrated]')
    PpL = PpR'

    # Measurement Cov
    @assert iszero(R)
    SL = PpL[I1, :]
    S = SL*SL'

    # Update
    Sinv = inv(S)
    K = PpL * (PpL[I1, :])' * Sinv  # P_p * H' * inv(S)
    m_f = m_p .+ K * (0 .- z)
    PfL = PpL - K*PpL[I1, :]  # P_f = P_p - K*S*K'
    P_f = SquarerootMatrix(PfL)

    x_filt = Gaussian(m_f, P_f)
    integ.u = m_f[I0]

    # Estimate error for adaptive steps
    if integ.opts.adaptive

        err_est_unscaled = sqrt.(σ².*diag(HQH))

        err = DiffEqBase.calculate_residuals(
            dt * err_est_unscaled, integ.u, integ.uprev,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(err, t) # scalar

        if integ.EEst < one(integ.EEst)
            integ.cache.x = x_filt
        end
    end
end


function OrdinaryDiffEq.initialize!(integ, cache::Union{FastEK0ConstantCache, FastEK0Cache})
    @assert integ.saveiter == 1
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj*cache.x)
end
