function preconditioner(d, q)
    P_preallocated = Diagonal(zeros(d*(q+1)))

    @fastmath @inbounds function P(h)
        @simd for i in 1:d
            @simd for j in 0:q
                P_preallocated[j*d + i,j*d + i] = h^(j-q-1/2)
            end
        end
        return P_preallocated
    end

    return P
end
