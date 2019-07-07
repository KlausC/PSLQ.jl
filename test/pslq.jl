import PSLQ: make_H, make_D, pslq_step, check_H, findmaxj, lqrefresh, integertype

function example_1(a, n::Integer)
    x = Vector{eltype(a)}(undef, n)
    b = copy(a)
    for i = 1:n-1
        x[i] = b
        b *= a
    end
    x[n] = sum(x[1:n-1] .* (1:n-1))
    x
end

function example_2(a, n::Integer)
    x = Vector{eltype(a)}(undef, n)
    b = oftype(a, 1)
    for i = 1:n
        x[i] = b
        b *= a
    end
    x
end

function make_Htest(n::Integer; gamma=1, T=Float64)
    H = zeros(T, n, n-1)
    for i = 1:n-1; H[i,i] = gamma^(1-i); end
    for i = 1:n, j = 1:i-1
        H[i,j] = i + j*j
    end
    H
end

@testset "correctness of calculations" begin
   
    H = make_Htest(1)
    @test_throws DimensionMismatch check_H(H)

    @test integertype(Float32) == Int
    @test integertype(Float64) == Int
    @test integertype(BigFloat) == BigInt

    H = make_Htest(10, gamma=2.0)
    @test findmaxj(H, 2.0) == 9
    H[1,1] = nextfloat(H[1,1])
    @test findmaxj(H, 2.0) == 1
    H[7,7] = nextfloat(H[7,7])
    @test findmaxj(H, 2.0) == 7

    H = make_Htest(10)
    D = make_D(H)
    @test norm(D * H - Matrix(I, size(H))) == 0

    x = example_1(float(ℯ), 6)
    H = make_H(x)
    P = I - x * x' / norm(x)^2
    @test norm(H * H' - P) <= 2*eps()*sqrt(size(H,2))
   
    D = make_D(H)
    @test D \ D == I

    @test_throws DimensionMismatch pslq_step(zeros(4,2), zeros(Int,4,4), zeros(Int,4,4))
    @test_throws DimensionMismatch pslq_step(zeros(4,3), zeros(Int,4,3), zeros(Int,4,3))

    A1 = Matrix(I*1, size(H, 1), size(H,1))
    B1 = copy(A1)
    H1 = H
    H1, A1, B1 = pslq_step(H1, A1, B1)
    H1, A1, B1 = pslq_step(H1, A1, B1)
    H1, A1, B1 = pslq_step(H1, A1, B1)
    H1, A1, B1 = pslq_step(H1, A1, B1)
    @test A1 * B1 == I
    X = H \ (B1 * H1) # verify that H = B1*H1 * X with orthonormal X
    @test norm(X'X - I) <= 10*eps()*size(H,2)
end

@testset "iteration algorithm with type $T" for T in (Float64,BigFloat)

    x = example_2(T(2.0)^(T(1)/7), 8)
    r = pslq(x)
    @test r.code == 1
    @test r.iter >= 30
    @test r.col == 1
    @test norm((r.B'x)[r.col]) <= 10*eps(norm(x))

    H = r.H
    H0 = make_H(x)
    H2 = lqrefresh(r.A, H0)
    @test norm(H2 - H) <= 1e8 * eps(norm(x))
end

