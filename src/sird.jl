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
    j = findmaxj(H, γ)
    if j < nt
        k = j+1
        givenscol!(H, j, k, err) # do nothing, if n >= nt; H = H * Q
    else
        v, k = findmax(abs, view(H, (nt+1):n, nt))
        k += nt
        if v < abs(H[j, j]) * sqrt(eps(real(T))) # TODO change to new parameter in err?
            H[(nt+1):n, nt] .= zero(T)
        else
            s = conj(sign(H[k, nt]))
            H[nt:n, nt] .*= s
        end
    end
    exrows!(H, j, k, 1:min(k, nt))
    exrows!(E, j, k, 1:min(k, nt))
    excols!(B, j, k, 1:n)
    make_DH!(D, H, B, err, 1) # H .= D * H, B = B / D
    H, B
end

"""
    findmaxj(H, γ)

On the diagonal of H find a maximal H[i] * γ^i

findmax(abs.(diag(H) .* γ .^ (1:n-1)))[2]
"""
function findmaxj(H::Matrix, γ::Real)
    # sqrt(3/4 - 1/γ^2) must be > 0; γ > 2 / √3
    # j = findmax(abs.(diag(H) .* γ .^ (1:n-1)))[2]
    n = size(H, 2)
    hnn = abs2(H[n, n])
    msf = hnn * γ^2
    for k = (n-1):-1:1
        hkk = abs2(H[k, k])
        if hkk >= msf && hkk > hnn + abs2(H[k+1, k])
            n, msf = k, hkk
        end
        msf *= γ
        hnn = hkk
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

function nint(x::T, y::T) where T<:Real
    IT = integertype(T)
    if abs(x)*2 <= abs(y)
        zero(IT)
    else
        iszero(y) ? y : IT(round(x / y))
    end
end

function nint(x::T, y::T) where T<:Complex
    integertype(T)(round(x / y))
end
integertype(::Type{T}) where T<:Real = typeof(Integer(real(T)(0)))
integertype(::Type{T}) where T<:Complex = Complex{integertype(real(T))}
integertype(::Type{BigFloat}) = BigInt

# calculate D and D*H in one loop (modified Hermite reduction), also B /= D
function make_DH!(DU::UnitLowerTriangular, H::AbstractMatrix, B::AbstractMatrix, err::ErrorEstimation, start::Int)
    n, t = check_H(H)
    nt = n - t
    E = err.E
    for i = (start+1):n
        for j = 1:(i-1)
            DU[i, j] = 0
        end
        j0 = start
        for j = min(i-1, nt):-1:1
            j < j0 && break
            hjj = H[j, j]
            q = iszero(hjj) ? zero(eltype(B)) : nint(H[i, j], hjj) # round to next integer
            if !iszero(q)
                j0 = 1
                for k = 1:j
                    DU[i, k] -= q * DU[j, k]
                    H[i, k] -= q * H[j, k]
                    E[i, k] += abs(q) * E[j, k]
                    #clean_H!(err, i, k)
                end
            end
        end
    end
    rdiv!(B, DU)
    DU, H, B
end

function make_H(X::AbstractVecOrMat{T}) where T<:Number
    n, t = size(X, 1), size(X, 2)
    nt = n - t
    Y = float([X I[1:n, 1:(n-t)]])
    Q, R = qr(Y)
    logabsdet(R)[1] > log(eps(real(eltype(R)))) * 0.9 || throw(ArgumentError("Matrix is singular"))
    H = Q[:, (t+1):n]
    for j = 2:nt, i = 1:(j-1)
        H[i, j] = 0
    end
    for j = 1:nt
        H[j, j] = Complex(real(H[j, j]))
    end
    H
end

function givenscol!(H::Matrix, j::Integer, k::Integer, err::ErrorEstimation)
    n, nt = size(H)
    if k > nt
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
    H[j, j], H[j, k] = a * conj(b), -a * c
    E[j, j] = E[j, k] = E[k, k] + E[j, j]
    #clean_H!(err, k, j); clean_H!(err, j, j);clean_H!(err, j, k)
    for i = (k+1):n
        u, v = H[i, j], H[i, k]
        H[i, j], H[i, k] = u * conj(b) + v * conj(c), -u * c + v * b
        E[i, j] = E[i, k] = E[i, j] + E[i, k]
        #clean_H!(err, i, j); clean_H!(err, i, k)
    end
    Q = LinearAlgebra.Givens(j, k, (b), -(c))
    H
end

function printx(text, B, H)
    BH = B * H
    println(text, " norm((B*H)'*(B*H) - I): ", norm(BH'BH - I))
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

function solution_column(sr::SirdResult)
    B = sr.B
    k = sr.col
    if sr.code == SUCCESS
        c = conj(view(B, :, k))
        i = findfirst(x -> abs2(x) == 1, c)
        if i !== nothing
            ci = c[i]
            if ci != 1
                c .*= conj(ci)
            end
        end
        c
    else
        vec(B[1:0])
    end
end
