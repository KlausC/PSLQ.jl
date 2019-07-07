module PSLQ

export pslq

using LinearAlgebra

function pslq(x::Vector{T}; criteria=nothing) where T<:Real
    n = length(x)
    H = make_H(x)
    criteria = make_criteria(criteria, x)
    A = Matrix(I*integertype(T)(1), n, n)
    B = copy(A)
    D = UnitLowerTriangular(similar(B)) # working area
    iter = 0
    result = (code=0, result=similar(B, n, 0))
    while iter < criteria.itermax && result.code == 0
        iter += 1    
        pslq_step!(H, A, B, D)
        result = convergence(x, H, A, B, iter, criteria)
    end
    result
end


function convergence(x::Vector{T}, H::AbstractMatrix, A::Matrix, B::Matrix,
                     iter::Integer, criteria::NamedTuple) where T
    n = size(x, 1)
    function xtb(j)
        s1 = s2 = T(0)
        for k = 1:n
            bkjx = B[k,j] * x[k]
            s1 += bkjx
            s2 += abs(bkjx)
        end
        s1, s2
    end
    resvec = []
    for j = 1:n
        s1, s2 = xtb(j)
        if abs(s1) <= s2 * criteria.rtol + criteria.atol
            push!(resvec, (code=1, col=j, xb=s1, xc=s2))
        end
    end
    if length(resvec) >= 1
        r = resvec[1]
        return (code=r.code, iter=iter, col=r.col, xb=r.xb, xc=r.xc, A=A, B=B, H=H)
    end
    (code=0, iter=iter, A=A, B=B, H=H)
end

function make_criteria(criteria, x)
    return (rtol=10*eps(eltype(x)), atol=10*eps(norm(x)), itermax=10000)
end
"""
    H1, A1, B1 = pslq_step(H, A, B)

Perform one step of the PSLQ agorithm. Before the first step H should be initialized
by calling `make_H(x)` from and array `x` of reals and `A`, `B` should be identity matriices.
The step returns modified `H` and `B` accordingly. The follow identities are maintained:
`A1 * B1 == A * B` and `B1 * H1 == B * H * G` where `G` is orthonormal.
"""
function pslq_step(H::Matrix{T}, A::Matrix{Ti}, B::Matrix{Ti}) where {T<:Real,Ti<:Integer}
    pslq_step!(copy(H), copy(A), copy(B), UnitLowerTriangular(similar(B)))
end
function pslq_step!(H::Matrix{T}, A::Matrix{Ti}, B::Matrix{Ti},
                    D::UnitLowerTriangular{Ti,Matrix{Ti}}) where {T<:Real,Ti<:Integer}
    n = check_H(H)
    check_square(D, n)
    check_square(B, n)
    make_D!(D, H, nint)
    lmul!(D, H)
    γ = 2.0
    j = findmaxj(H, γ)
    exrows!(H, j, j+1, 1:j)
    givenscol!(H, j)
    rdiv!(B, D)
    excols!(B, j, j+1, 1:n)
    lmul!(D, A)
    exrows!(A, j, j+1, 1:n)
    H, A, B
end

function findmaxj(H::Matrix, γ::Real)
    # δ = sqrt(3/4 - 1/γ^2) must be > 0
    # j = findmax(abs.(diag(H) .* γ .^ (1:n-1)))[2]
    n = min(size(H)...)
    j = 1
    msf = abs(H[1,1]) / γ
    for k = 2:n
        hkk = abs(H[k,k])
        if hkk >= msf
            j, msf = k, hkk
        end
        msf /= γ
    end
    j
end
function check_H(H::AbstractMatrix)
    n = size(H, 1)
    if size(H, 2) != n - 1 || n < 2
        throw(DimensionMismatch("Matrix needs to be nx(n-1) with n >= 2 " *
                                "but has size $(size(H))"))
    end
    n
end
function check_square(B::AbstractMatrix, n::Integer)
    n = size(B, 1)
    if size(B, 2) != n
        throw(DimensionMismatch("Matrix needs to be square with size $n but is $(size(B))"))
    end
    n
end
nint(x) = Integer(round(x))
integertype(::Type{T}) where T = typeof(Integer(T(0)))
integertype(::Type{BigFloat}) = BigInt
function make_D(H::Matrix{T}, nint::Function=nint) where {T<:Real}
    n = size(H, 1)
    D = Matrix{integertype(T)}(undef, n, n)
    make_D!(UnitLowerTriangular(D), H, nint)
end
function make_D!(DU::UnitLowerTriangular, H::Matrix, nint::Function)
    n = check_H(H)
    D = DU.data
    for i = 2:n
        for j = i-1:-1:1
            sum = H[i,j]
            for k = j+1:i-1
                sum += D[i,k] * H[k,j]
            end
            D[i,j] = nint(-sum / H[j,j])
        end
    end
    DU
end

function make_H(x::Vector{T}) where T<:Real
    n = length(x)
    H = zeros(T, n, n-1)
    b = abs(x[n])
    a = b^2
    for j = n-1:-1:1
        a += x[j]^2
        c = b
        b = sqrt(a)
        H[j,j] = c / b
        d = -x[j] / (c * b)
        for i = j+1:n
            H[i,j] = x[i] * d
        end
    end
    H
end

function givenscol!(H::Matrix{T}, j::Integer) where T<:Real
    if j == size(H, 2)
        return refresh!(H, j)
    end
    n = size(H, 1)
    a, b, c = H[j,j], H[j+1,j+1], H[j+1,j]
    d = hypot(a, b)
    a /= d
    b /= d
    H[j,j] = d
    H[j+1,j], H[j+1,j+1] = c * a, -c * b
    for k = j+2:n
        u, v = H[k,j], H[k,j+1]
        H[k,j], H[k,j+1] = u*a + v*b, -u*b + v*a
    end
    refresh!(H, j+1)
end

function exrows!(H::Matrix, i1::Integer, i2::Integer, r::UnitRange)
    for k in r
        H[i1,k], H[i2,k] = H[i2,k], H[i1,k]
    end
    H
end

function excols!(H::Matrix, i1::Integer, i2::Integer, r::UnitRange)
    for k in r
        H[k,i1], H[k,i2] = H[k,i2], H[k,i1]
    end
    H
end

"""
    refresh!(H, j)
divide column `j` by `H[j,j]/abs(H[j,j])` in order to make `H[j,j]` real and positive. 
`H` is assumed to be lower trapezoidal.
"""
function refresh!(H::Matrix, col::Integer)
    m, n = size(H)
    h = H[col, col]
    if 0 < col <= m && !(isreal(h) && real(h) >= 0)
        h /= abs(h)
        @inbounds for k = col:m
            H[k,col] /= h
        end
    end
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
    L = Matrix(qr!(H'*A').R')
    refresh!(L)
end

end # module
