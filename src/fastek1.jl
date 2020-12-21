Base.@kwdef struct FastEK1 <: AbstractEK
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end


mutable struct FastEK1ConstantCache{
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
    alg::FastEK1, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP::Val{false})

    q = alg.order
    u0 = u
    t0 = t
    d = length(u)

    Proj(deriv) = kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    SolProj = Proj(0)

    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    @assert alg.diffusionmodel == :dynamic "FastEK1 only uses dynamic diffusion"

    Precond = preconditioner(d, q)
    _, Q = ibm(d, q, uEltypeNoUnits)
    A, _ = vanilla_ibm(d, q, uEltypeNoUnits)
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
    P0 = SquarerootMatrix(P0)
    # P0 = SquarerootMatrix(P0)
    x0 = Gaussian(m0, P0)

    initdiff = one(uEltypeNoUnits)

    # I0, I1 = SVector{d}(1:d), SVector{d}(d+1:2d)
    I0, I1 = 1:d, d+1:2d

    return FastEK1ConstantCache{
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


function OrdinaryDiffEq.perform_step!(integ, cache::FastEK1ConstantCache, repeat_step=false)
    @unpack t, dt, f, p = integ
    @unpack d, q, Proj, SolProj, Precond, I0, I1 = integ.cache
    @unpack x, A, Q, R = integ.cache
    D = d*(q+1)

    tnew = t + dt

    # Setup
    TI = inv(Precond(dt))
    Ah = A(dt)
    Qh = TI*Q*TI
    # HQH = (TI*Q*TI)[I1]

    m, P = x.μ, x.Σ
    PL = P.squareroot

    # Predict - Mean
    # m_p = Ah*m
    m_p = Ah*m

    u_pred = @view m_p[I0]  # E0 * m_p
    du_pred = @view m_p[I1]  # E1 * m_p

    # Measure
    du = f(u_pred, p, tnew)
    z_neg = du - du_pred

    # Measure Jac and build H
    ddu = f.jac(u_pred, p, t)
    E0, E1 = Proj(0), Proj(1)
    H = (E1 - ddu*E0)

    # Calibrate
    # σ² = z' * (HQH \ z) / d  # z'inv(H*Q*H')z / d
    σ² = z_neg'*((H*Q*H')\z_neg) / d
    cache.diffusion = σ²

    # Predict - Cov
    QhL_calibrated = sqrt(σ²) .* TI.diag .* Q.L
    small_qr_input = [Ah*PL QhL_calibrated]'
    Pp = small_qr_input'small_qr_input
    chol = cholesky(Pp, check=false)
    PpL = issuccess(chol) ? chol.U' : qr(small_qr_input).R'

    # Measurement Cov
    # @assert iszero(R)
    S = H*Pp*H'

    # Update
    Sinv = inv(S)
    # K = PpL * PpL[2, :] * Sinv  # P_p * H' * inv(S)
    K = Pp*H'* Sinv  # P_p * H' * inv(S)
    m_f = m_p .+ K*z_neg
    PfL = PpL - K*H*PpL
    P_f = SquarerootMatrix(PfL)

    x_filt = Gaussian(m_f, P_f)
    integ.u = m_f[I0]

    # Estimate error for adaptive steps
    if integ.opts.adaptive

        err_est_unscaled = sqrt.(σ²*diag(H*Q*H'))

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


function OrdinaryDiffEq.initialize!(integ, cache::Union{FastEK1ConstantCache})
    @assert integ.saveiter == 1
    if integ.opts.dense
        OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, cache.x)
    end
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj*cache.x)
end
