abstract type AbstractDiffusion end
abstract type AbstractStaticDiffusion <: AbstractDiffusion end
abstract type AbstractDynamicDiffusion <: AbstractDiffusion end
isstatic(diffusion::AbstractStaticDiffusion) = true
isdynamic(diffusion::AbstractStaticDiffusion) = false
isstatic(diffusion::AbstractDynamicDiffusion) = false
isdynamic(diffusion::AbstractDynamicDiffusion) = true
initial_diffusion(diffusion::AbstractDiffusion, d, q, Eltype) = one(Eltype)


struct FixedDiffusion <: AbstractStaticDiffusion end
function estimate_diffusion(rule::FixedDiffusion, integ)
    @unpack d, measurement = integ.cache
    diffusions = integ.sol.diffusions

    v, S = measurement.μ, measurement.Σ

    if iszero(v)
        return zero(integ.cache.diffusion)
    end
    if iszero(S)
        return Inf
    end

    diffusion_t = v' * inv(S) * v / d

    if integ.success_iter == 0
        @assert length(diffusions) == 0
        return diffusion_t
    else
        @assert length(diffusions) == integ.success_iter
        diffusion_prev = diffusions[end]
        diffusion = diffusion_prev + (diffusion_t - diffusion_prev) / integ.success_iter
        return diffusion
    end
end


"""Maximum a-posteriori Diffusion estimate when using an InverseGamma(1/2,1/2) prior

The mode of an InverseGamma(α,β) distribution is given by β/(α+1)
To compute this in an on-line basis from the previous Diffusion, we reverse the computation to
get the previous sum of residuals from Diffusion, and then modify that sum and compute the new
Diffusion.
"""
struct MAPFixedDiffusion <: AbstractStaticDiffusion end
function estimate_diffusion(rule::MAPFixedDiffusion, integ)
    @unpack d, measurement = integ.cache
    diffusions = integ.sol.diffusions

    N = integ.success_iter + 1
    v, S = measurement.μ, measurement.Σ
    res_t = v' * inv(S) * v / d

    α, β = 1/2, 1/2
    if integ.success_iter == 0
        @assert length(diffusions) == 0
        diffusion_t = (β + 1/2 * res_t) / (α + N*d/2 + 1)
        return diffusion_t
    else
        @assert length(diffusions) == integ.success_iter
        diffusion_prev = diffusions[end]
        res_prev = (diffusion_prev * (α + (N-1)*d/2 + 1) - β) * 2
        res_sum_t = res_prev + res_t
        diffusion = (β + 1/2 * res_sum_t) / (α + N*d/2 + 1)
        return diffusion
    end
end


struct DynamicDiffusion <: AbstractDynamicDiffusion end
function estimate_diffusion(kind::DynamicDiffusion, integ)
    @unpack dt = integ
    @unpack d, R = integ.cache
    @unpack H, Q, measurement = integ.cache
    # @assert all(R .== 0) "The dynamic-diffusion assumes R==0!"
    z = measurement.μ
    σ² = z' * ((H*Q*H')\z) / d
    return σ²
end


struct MVDynamicDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(diffusion::MVDynamicDiffusion, d, q, Eltype) =
    kron(Diagonal(ones(Eltype, q+1)), Diagonal(ones(Eltype, d)))
function estimate_diffusion(kind::MVDynamicDiffusion, integ)
    @unpack dt = integ
    @unpack d, q, R, Precond, Proj = integ.cache
    @unpack H, Q, measurement = integ.cache
    E1 = Proj(1)
    z = measurement.μ

    # @assert all(R .== 0) "The dynamic-diffusion assumes R==0!"

    # Assert EK0
    PI = inv(Precond(dt))
    @assert all(H .== E1 * PI)

    # More safety checks
    @assert isdiag(H*Q*H')
    @assert length(unique(diag(H*Q*H'))) == 1
    Q0_11 = diag(H*Q*H')[1]

    Σ_ii = z .^ 2 ./ Q0_11
    Σ_ii .= max.(Σ_ii, eps(eltype(Σ_ii)))
    Σ = Diagonal(Σ_ii)

    Σ_out = kron(Diagonal(ones(q+1)), Σ)
    @assert isdiag(Σ_out)
    # @info "MVDynamic diffusion" Σ Σ_out
    return Σ_out
end


struct MVFixedDiffusion <: AbstractStaticDiffusion end
initial_diffusion(diffusion::MVFixedDiffusion, d, q, Eltype) =
    kron(Diagonal(ones(Eltype, q+1)), Diagonal(ones(Eltype, d)))
function estimate_diffusion(kind::MVFixedDiffusion, integ)
    @unpack dt = integ
    @unpack d, q, R, Precond, Proj = integ.cache
    @unpack measurement, H = integ.cache
    @unpack d, measurement = integ.cache
    E1 = Proj(1)
    diffusions = integ.sol.diffusions

    # Assert EK0
    PI = inv(Precond(dt))
    @assert all(H .== E1 * PI)

    @unpack measurement = integ.cache
    v, S = measurement.μ, measurement.Σ

    # More safety checks
    @assert isdiag(S)
    # @assert length(unique(diag(S))) == 1
    S_11 = diag(S)[1]

    Σ_ii = v .^ 2 ./ S_11
    Σ = Diagonal(Σ_ii)
    Σ_out = kron(Diagonal(ones(q+1)), Σ)
    @assert isdiag(Σ_out)
    # @info "MV-MLE-Diffusion" v S Σ Σ_out

    if integ.success_iter == 0
        @assert length(diffusions) == 0
        return Σ_out
    else
        @assert length(diffusions) == integ.success_iter
        diffusion_prev = diffusions[end]
        diffusion = diffusion_prev + (Σ_out - diffusion_prev) / integ.success_iter
        return diffusion
    end
end
