abstract type AbstractKroneckerMatrix{T<:Real} <: AbstractMatrix{T} end
struct KroneckerMatrix{T<:Real, L<:AbstractMatrix, R<:AbstractMatrix} <: AbstractKroneckerMatrix{T}
    left::L
    right::R
end
KroneckerMatrix(M::AbstractMatrix, d::Int) = KroneckerMatrix{eltype(M), typeof(M), typeof(I(d))}(M, I(d))
Base.size(K::KroneckerMatrix) = Base.size(K.left) .* Base.size(K.right)
Base.Matrix(K::KroneckerMatrix) = kron(K.left, K.right)
Base.getindex(K::KroneckerMatrix, I::Vararg{Int, N}) where {N} = getindex(Matrix(K), I...)
Base.copy(K::KroneckerMatrix{T, L, R}) where {T, L, R} = KroneckerMatrix{T, L, R}(copy(K.left), copy(K.right))
Base.copy!(dst::KroneckerMatrix, src::KroneckerMatrix) =
    (Base.copy!(dst.left, src.left); Base.copy!(dst.right, dst.right); nothing)



struct KronMat{T<:Real, L<:AbstractMatrix{T}, I<:Int} <: AbstractKroneckerMatrix{T}
    left::L
    rightd::I
end
Base.size(K::KronMat) = Base.size(K.left) .* K.rightd
Base.Matrix(K::KronMat) =
    (
        @warn "called Matrix on KronMat!";
        kron(K.left, I(K.rightd)))
Base.getindex(K::KronMat, I::Vararg{Int, N}) where {N} =
    (
        @warn "called getindex on KronMat!";
        getindex(Matrix(K), I...))
Base.copy(K::KronMat{T, L, R}) where {T, L, R} =
    KronMat{T, L, R}(copy(K.left), K.rightd)
Base.copy!(dst::KronMat, src::KronMat) =
    (@assert dst.rightd == src.rightd; Base.copy!(dst.left, src.left); nothing)

Base.:*(K1::KronMat, K2::KronMat) = (
    @assert K1.rightd == K2.rightd;
    KronMat(K1.left*K2.left, K1.rightd)
)
Base.:*(K::KronMat, m::AbstractVector) =
    (d = K.rightd; return vec(reshape(m, (d, :)) * K.left'))
