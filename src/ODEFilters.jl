__precompile__()

module ODEFilters

using Reexport
@reexport using DiffEqBase
using OrdinaryDiffEq
import DiffEqBase: check_error!

import Base: copy, copy!, show, size, ndims
stack(x) = copy(reduce(hcat, x)')

using LinearAlgebra
using TaylorSeries
@reexport using StructArrays
using UnPack
using RecipesBase
using ModelingToolkit
using RecursiveArrayTools
using StaticArrays


@reexport using PSDMatrices
import PSDMatrices: X_A_Xt
X_A_Xt(A, X) = X*A*X'
apply_diffusion(Q, diffmat::Diagonal) = X_A_Xt(Q, sqrt.(diffmat))
apply_diffusion(Q, diffmat::Number) = Q*diffmat


# All the Gaussian things
@reexport using GaussianDistributions
using GaussianDistributions: logpdf
const PSDGaussian{T} = Gaussian{Vector{T}, PSDMatrix{T}}
const PSDGaussianList{T} = StructArray{PSDGaussian{T}}
copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
copy!(dst::Gaussian, src::Gaussian) = (copy!(dst.μ, src.μ); copy!(dst.Σ, src.Σ); nothing)
show(io::IO, g::Gaussian) = print(io, "Gaussian($(g.μ), $(g.Σ))")
show(io::IO, ::MIME"text/plain", g::Gaussian{T, S}) where {T, S} =
    print(io, "Gaussian{$T,$S}($(g.μ), $(g.Σ))")
size(g::Gaussian) = size(g.μ)
ndims(g::Gaussian) = ndims(g.μ)

Base.:*(M, g::PSDGaussian) = Gaussian(M * g.μ, X_A_Xt(g.Σ, M))
GaussianDistributions.whiten(Σ::PSDMatrix, z) = Σ.L\z

import Statistics: mean, var, std
var(p::PSDGaussian{T}) where {T} = diag(p.Σ)
std(p::PSDGaussian{T}) where {T} = sqrt.(diag(p.Σ))
mean(s::PSDGaussianList{T}) where {T} = mean.(s)
var(s::PSDGaussianList{T}) where {T} = var.(s)
std(s::PSDGaussianList{T}) where {T} = std.(s)

include("priors.jl")
include("diffusions.jl")

include("algorithms.jl")
export EK0, EK1
include("alg_utils.jl")
include("caches.jl")
include("state_initialization.jl")
include("integrator_utils.jl")
include("filtering.jl")
include("perform_step.jl")
include("preconditioning.jl")
include("smoothing.jl")
include("solution.jl")

# Utils
include("jacobian.jl")
include("numerics_tricks.jl")

# Iterated Extended Kalman Smoother
include("ieks.jl")
export IEKS, solve_ieks

end
