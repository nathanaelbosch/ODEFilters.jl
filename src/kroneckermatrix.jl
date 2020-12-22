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
        @warn "called getindex on KronMat!" I;
        error("This is not nice");
        getindex(Matrix(K), I...))
Base.copy(K::KronMat{T, L, R}) where {T, L, R} =
    KronMat{T, L, R}(K.left isa LinearAlgebra.Adjoint ? copy(K.left')' : copy(K.left), K.rightd)
Base.copy!(dst::KronMat, src::KronMat) =
    (@assert dst.rightd == src.rightd; Base.copy!(dst.left, src.left); nothing)

Base.:*(K1::KronMat, K2::KronMat) = (
    @assert K1.rightd == K2.rightd;
    KronMat(K1.left*K2.left, K1.rightd)
)
Base.:*(K::KronMat, m::AbstractVector) =
    (d = K.rightd; return vec(reshape(m, (d, :)) * K.left'))
