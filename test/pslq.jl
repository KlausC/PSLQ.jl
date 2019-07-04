
function example_1(a, n::Integer)
    x = Vector{eltype(a)}(undef, n)
    b = copy(a)
    for i = 1:n-1
        x[i] = b
        b *= a
    end
    x[n] = sum(x[1:n-1] .* (1:n-1))
    x ./= norm(x)
    x
end

function make_Htest(n::Integer, T=Float64)
    H = zeros(T, n, n-1)
    for i = 1:n-1; H[i,i] = 1; end
    for i = 1:n, j = 1:i-1
        H[i,j] = i + j*j
    end
    H
end

import PSLQ: make_H, make_D, make_E, make_G, make_R, pslq_step

@testset "correctness of calculations" begin
    
    H = make_Htest(10)
    D = make_D(H)
    @test norm(D * H - Matrix(I, size(H))) == 0

    x = example_1(float(ℯ), 6)
    H = make_H(x)
    P = I - x * x'
    @test norm(H * H' - P) <= 2*eps()*sqrt(size(H,2))
   
    D = make_D(H)
    E = make_E(D)
    @test norm(D * E - I) == 0

    A1 = Matrix(I, size(H, 1), size(H,1))
    B1 = copy(A1)
    H1 = H
    H1, A1, B1 = pslq_step(H1, A1, B1)
    H1, A1, B1 = pslq_step(H1, A1, B1)
    H1, A1, B1 = pslq_step(H1, A1, B1)
    H1, A1, B1 = pslq_step(H1, A1, B1)
    X = H \ (B1 * H1) # verify that H = B1*H1 * X with orthogonal X
    @test norm(X'X - I) <= 10*eps()*size(H,2)
end

