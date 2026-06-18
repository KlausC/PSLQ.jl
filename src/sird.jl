"""
    H1, A1, B1 = pslq_step(H, A, B)

Perform one step of the PSLQ agorithm. Before the first step H should be initialized
by calling `make_H(X)` from nxt-array `X` of reals and `A`, `B` should be identity matrices.
The step returns modified `H` `A`,and `B` accordingly.
The follow identities are maintained:
`A1 * B1 == A * B` and `B1 * H1 == B * H * G` where `G` is orthonormal.
"""
function sird_step!(γ::Real, H::Matrix{T}, B::Matrix{Ti}, D::AbstractMatrix{Ti}, err::ErrorEstimation) where {T,Ti}
    n, t = check_H(H)
    nt = n - t
    E = err.E
    check_square(D, n)
    check_square(B, n)
    make_DH!(D, H, B, err) # H .= D * H, B = B / D
    j = findmaxj(H, γ)
    k = j < nt ? j+1 : findmax(abs, view(H, (nt+1):n, nt))[2] + nt
    givenscol!(H, j, k, err) # do nothing, if n >= nt
    exrows!(H, j, k, 1:min(k, nt))
    exrows!(E, j, k, 1:min(k, nt))
    excols!(B, j, k, 1:n)
    H, B
end

"""
    findmax(H, γ)

On the diagonal of H find a maximal H[i] * γ^i

findmax(abs.(diag(H) .* γ .^ (1:n-1)))[2]
"""
function findmaxj(H::Matrix, γ::Real)
    # sqrt(3/4 - 1/γ^2) must be > 0; γ > 2 / √3
    # j = findmax(abs.(diag(H) .* γ .^ (1:n-1)))[2]
    n = min(size(H)...)
    msf = abs(H[n, n]) * γ
    for k = (n-1):-1:1
        hkk = abs(H[k, k])
        if hkk >= msf
            n, msf = k, hkk
        end
        msf *= γ
    end
    n
end
function check_H(H::AbstractMatrix)
    n, nt = size(H, 1), size(H, 2)
    t = n - nt
    if t <= 0 || nt <= 0
        throw(DimensionMismatch("Matrix needs to be nx(n-t) with 0 < t < n" *
                                "but has size $(size(H))"))
    end
    n, t
end
function check_square(B::AbstractMatrix, n::Integer)
    m = size(B, 1)
    if size(B, 2) != n || m != n
        throw(DimensionMismatch("Matrix needs to be square with size $n but is $(size(B))"))
    end
    n
end
nint(x::T, y::T) where T<:Real = iszero(y) ? y : integertype(T)(round(x / y))

integertype(::Type{T}) where T<:Real = typeof(Integer(zero(T)))

# calculate D and D*H in one loop (modified Hermite reduction)
function make_DH!(DU::UnitLowerTriangular, H::AbstractMatrix, B::AbstractMatrix, err::ErrorEstimation)
    n, t = check_H(H)
    nt = n - t
    E = err.E
    for i = 2:n
        for j = 1:(i-1)
            DU[i, j] = 0
        end
        for j = min(i-1, nt):-1:1
            q = nint(H[i, j], H[j, j]) # round to next integer
            for k = 1:j
                DU[i, k] -= q * DU[j, k]
                H[i, k] -= q * H[j, k]
                E[i, k] += abs(q) * E[j, k]
            end
        end
    end
    rdiv!(B, DU)
    DU, H, B
end

function make_H(X::AbstractVecOrMat{T}) where T<:Real
    n, t = size(X, 1), size(X, 2)
    nt = n - t
    Y = float([X I[1:n, 1:(n-t)]])
    Q, R = qr(Y)
    logabsdet(R)[1] > log(eps(eltype(R))) * 0.9 || throw(ArgumentError("Matrix is singular"))
    H = Q[:, (t+1):n]
    for j = 2:nt, i = 1:(j-1)
        H[i, j] = 0
    end
    H
end

function givenscol!(H::Matrix, j::Integer, k::Integer, err::ErrorEstimation)
    if k > size(H, 2)
        return H
    end
    E = err.E
    n = size(H, 1)
    a, b, c = H[j, j], H[k, j], H[k, k]
    d = hypot(b, c)
    b /= d
    c /= d
    H[k, j], H[k, k] = d, 0
    E[k, j], E[k, k] = E[k, j] + E[k, k], 0
    H[j, j], H[j, k] = a * b, -a * c
    E[j, j] = E[j, k] = E[k, k] + E[j, j]
    for i = (k+1):n
        u, v = H[i, j], H[i, k]
        H[i, j], H[i, k] = u * b + v * c, -u * c + v * b
        E[i, j] = E[i, k] = E[i, j] + E[i, k]
    end
    H
end

function exrows!(H::Matrix, i1::Integer, i2::Integer, r::UnitRange)
    for k in r
        H[i1, k], H[i2, k] = H[i2, k], H[i1, k]
    end
    H
end

function excols!(H::Matrix, i1::Integer, i2::Integer, r::UnitRange)
    for k in r
        H[k, i1], H[k, i2] = H[k, i2], H[k, i1]
    end
    H
end

"""
    refresh!(H, j)
divide column `j` by `H[j,j]/abs(H[j,j])` in order to make `H[j,j]` real and positive.
`H` is assumed to be lower trapezoidal.
"""
function refresh!(H::Matrix, col::Integer)
    H
end

function refresh!(H::Matrix)
    n = size(H, 2)
    @inbounds for i = 1:n
        refresh!(H, i)
    end
    H
end

"""
    lqrefresh(A, H)

calculate LQ-factorization of `A * H`and return refreshed `L`.
"""
function lqrefresh(A::Matrix, H::Matrix)
    L = lq!(A * H).L # lq! not availbale for BigFloat
    refresh!(L)
end
function lqrefresh(A::Matrix, H::Matrix{BigFloat})
    L = Matrix(qr!(H' * A').R')
    refresh!(L)
end
