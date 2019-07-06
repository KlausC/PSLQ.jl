module PSLQ

using LinearAlgebra

function pslq_step(H::Matrix{T}, B::Matrix{Ti}) where {T<:Real,Ti<:Integer}
    pslq_step!(copy(H), copy(B), UnitLowerTriangular(similar(B)))
end
function pslq_step!(H::Matrix{T}, B::Matrix{Ti},
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
    H, B
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
    j >= size(H, 2) && return H
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
    H
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

function make_G(H, j)
    n = size(H, 1)
    G = zeros(eltype(H), n-1, n-1)
    for i = 1:n-1; G[i,i] = eltype(H)(1); end
    if j != n-1
        b, c = H[j+1,j], H[j+1,j+1]
        d = hypot(b, c)
        b, c = b/d, c/d
        G[j,j] = G[j+1,j+1] = b
        G[j,j+1] = -c
        G[j+1,j] = c
    end
    G
end

function make_R(H, j)
    n = size(H, 1)
    R = zeros(Int, n, n)
    for i = 1:n; R[i,i] = 1; end
    R[j,j] = R[j+1,j+1] = 0
    R[j,j+1] = R[j+1,j] = 1
    R
end



end # module
