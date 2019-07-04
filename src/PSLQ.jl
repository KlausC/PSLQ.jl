module PSLQ

using LinearAlgebra

function pslq_step(H, A, B)
    n = size(H, 1)
    D = make_D(H)
    E = make_E(D)
    H = D * H
    γ = 2.0
    δ = sqrt(3/4 - 1/γ^2)
    j = findmax(abs.(diag(H) .* γ .^ (1:n-1)))[2]
    Rj = make_R(H, j)
    Gj = make_G(H, j)
    H = Rj * H * Gj
    if j < n-1; H[j,j+1] = 0; end
    A = Rj * D * A
    B = B * E * Rj
    H, A, B
end

function make_D(H::Array{<:Real,2})
    nint(x) = Integer(round(x))
    Ti = typeof(Integer(eltype(H)(0)))
    n = size(H, 1)
    D = zeros(Ti, n, n)
    for i = 1:n; D[i,i] = Ti(1); end
    for i = n:-1:1;
        for j = i-1:-1:1
            sum = eltype(H)(0)
            for k = j:i
                sum += D[i,k] * H[k,j]
            end
            D[i,j] = nint(-sum/H[j,j])
        end
    end
    D
end

 function make_E(D::Array{<:Integer,2})
    E = zeros(eltype(D), size(D))
    n = size(D, 1)
    for i = 1:n; E[i,i] = 1; end
    for j = n:-1:1
        for i = j+1:n
            sum = eltype(D)(0)
            for k = j:i
                sum += E[i,k] * D[k,j]
            end
            E[i,j] = -sum
        end
    end
    E
end

function make_H(x::Vector{<:Real})
    x = x ./ norm(x)
    n = length(x)
    s = sqrt.(reverse(cumsum(reverse(x.^2))))
    H = zeros(eltype(x), n, n-1)
    for i = 1:n-1
        H[i,i] = s[i+1] / s[i]
    end
    for i = 1:n, j = 1:i-1
        H[i,j] = -x[i]*x[j]/s[j]/s[j+1]
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
