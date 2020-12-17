abstract type AbstractSquarerootMatrix{T<:Real} <: AbstractMatrix{T} end
struct SquarerootMatrix{T<:Real, S<:AbstractMatrix} <: AbstractSquarerootMatrix{T}
    squareroot::S
end
SquarerootMatrix(M::AbstractMatrix{T}) where {T} = SquarerootMatrix{T, typeof(M)}(M)
Base.Matrix(M::SquarerootMatrix) = M.squareroot*M.squareroot'
Base.size(M::SquarerootMatrix) = (d = Base.size(M.squareroot)[1]; (d,d))
Base.getindex(M::SquarerootMatrix, I::Vararg{Int, N}) where {N} =
    getindex(Matrix(M), I...)
Base.copy(M::SquarerootMatrix) = SquarerootMatrix(copy(M.squareroot))


X_A_Xt(M::SquarerootMatrix, X::AbstractMatrix) =
    SquarerootMatrix(X*M.squareroot)
