"""
    result = pslq(x::Vector; criteria)

Perform PSLQ iterations for the vector x until termination criteria are fulfilled.
`result` always contains the fields: `code`, `iter`, `A`, `B`, `H`.
The returned result is a named tuple and may contain different result variants:

`result.code == 1`:

 A solution `m` was found `|m' * x| <= cr.atol + cr.rtol*|x|`
 with `m = result.B[:,result.col]`

`result.code == 2`:
    No solution `m` possible with `|m| <= cr.solutionmax`

`result.code == 3`:
    No decision after number of iterations exceeds `cr.itermax`

`result.code == 4`:
    Inaccuracy in calculation - retry with higher precision input data

The input criteria are transformed to object `cr` by method `make_criteria(criteria, x)`.
"""
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

"""
    result = convergence(x, H, A, B, iter, criteria)

Termination check and provide result for `pslq`.
The arguments `H`, `A`, `B` are as described in the algoritm paper.
`result.code == 0` means that iterations should continue.
"""
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

