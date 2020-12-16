########################################################################################
# Caches
########################################################################################
abstract type ODEFiltersCache <: OrdinaryDiffEq.OrdinaryDiffEqCache end
mutable struct GaussianODEFilterCache{
    RType, ProjType, SolProjType, FP, uType, xType, AType, QType, matType, diffusionType, diffModelType,
    measType, llType,
} <: ODEFiltersCache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    A::AType
    Q::QType
    diffusionmodel::diffModelType
    R::RType
    Proj::ProjType
    SolProj::SolProjType
    Precond::FP
    # Mutable stuff
    u::uType
    u_pred::uType
    u_filt::uType
    tmp::uType
    x::xType
    x_pred::xType
    x_filt::xType
    x_tmp::xType
    x_tmp2::xType
    measurement::measType
    H::matType
    du::uType
    ddu::matType
    K::matType
    G::matType
    covmatcache::matType
    diffusion::diffusionType
    err_tmp::uType
    log_likelihood::llType
end

function OrdinaryDiffEq.alg_cache(
    alg::GaussianODEFilter, u, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP)
    initialize_derivatives=true

    if !(u isa AbstractVector)
        error("Problems which are not scalar- or vector-valued (e.g. u0 is a scalar or a matrix) are currently not supported")
    end

    if (alg isa EK1 || alg isa IEKS) && isnothing(f.jac)
        error("""EK1 requires the Jacobian. To automatically generate it with ModelingToolkit.jl use ODEFilters.remake_prob_with_jac(prob).""")
    end

    q = alg.order
    u0 = u
    t0 = t
    d = length(u)

    uType = typeof(u0)
    uElType = eltype(u0)
    matType = Matrix{uElType}

    # Projections
    Proj(deriv) = kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    SolProj = Proj(0)

    # Prior dynamics
    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    Precond = preconditioner(d, q)
    A, Q = ibm(d, q, uElType)

    # Measurement model
    R = PSDMatrix(LowerTriangular(zeros(d, d)))
    # Initial states
    m0, P0 = initialize_derivatives ?
        initialize_with_derivatives(u0, f, p, t0, q) :
        initialize_without_derivatives(u0, f, p, t0, q)
    @assert iszero(P0)
    P0 = PSDMatrix(LowerTriangular(zero(P0)))
    x0 = Gaussian(m0, P0)

    # Pre-allocate a bunch of matrices
    h = Proj(0) * x0.μ
    H = copy(Proj(0))
    du = copy(u0)
    ddu = zeros(uElType, d, d)
    v, S = copy(h), copy(ddu)
    measurement = Gaussian(v, S)
    K = copy(H')
    G = copy(Matrix(P0))
    covmatcache = copy(G)

    diffusion_models = Dict(
        :dynamic => DynamicDiffusion(),
        :dynamicMV => MVDynamicDiffusion(),
        :fixed => FixedDiffusion(),
        :fixedMV => MVFixedDiffusion(),
        :fixedMAP => MAPFixedDiffusion(),
    )
    diffmodel = diffusion_models[alg.diffusionmodel]
    initdiff = initial_diffusion(diffmodel, d, q, uEltypeNoUnits)

    return GaussianODEFilterCache{
        typeof(R), typeof(Proj), typeof(SolProj), typeof(Precond),
        uType, typeof(x0), typeof(A), typeof(Q), matType, typeof(initdiff),
        typeof(diffmodel), typeof(measurement), uEltypeNoUnits,
    }(
        # Constants
        d, q, A, Q, diffmodel, R, Proj, SolProj, Precond,
        # Mutable stuff
        copy(u0), copy(u0), copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0), copy(x0), copy(x0),
        measurement,
        H, du, ddu, K, G, covmatcache, initdiff,
        copy(u0),
        zero(uEltypeNoUnits),
    )
end
