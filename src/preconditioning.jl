function preconditioner(d, q)
    P_preallocated = Diagonal(zeros(d*(q+1)))

    @fastmath @inbounds function P(h)
        @simd for i in 1:d
            val = h^(-q-1/2)
            @simd for j in 0:q
                # P_preallocated[j*d + i,j*d + i] = h^(j-q-1/2)
                P_preallocated[j*d + i,j*d + i] = val
                val *= h
            end
        end
        return P_preallocated
    end

    return P
end

function invpreconditioner(d, q)
    @assert d == 1
    P_preallocated = Diagonal(zeros(d*(q+1)))

    @fastmath function P(h)
        val = h^(q + 1/2)
        @inbounds for j in 1:q+1
            P_preallocated[j,j] = val
            val /= h
        end
        return P_preallocated
    end

    return P
end
