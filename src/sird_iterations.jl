"""
Article:
    Simultaneous Integer Relation Detection and Its an Application
    Chen Jing-wei, Feng Yong, Qin Xiao-lin, and Zhang Jing-zhong (2010)
Link: https://arxiv.org/pdf/0906.4917
"""

"""
    result = sird(X::Matrix; criteria)

Perform SIRD iterations for the Matrix X until termination criteria are fulfilled.
`result` always contains the fields: `code`, `iter`, `A`, `B`, `H`.
The returned result is a named tuple and may contain different result variants:

`result.code == 1`:
     A solution `m` was found `|m' * X| <= cr.atol + cr.rtol*|X|`
    with `m = result.B[:,result.col]`

`result.code == 2`:
    Inaccuracy in calculation - retry using higher precision

`result.code == 3`:
    No solution `m` possible with `|m| <= cr.elimit`

`result.code == 4`:
    No decision after number of iterations exceeds `cr.itermax`

`result.code == 0`:
    Iteration ongoing

The result has teh following fields:

-   code: see above
-   col: number of column in `B`, which contains solution (if code == 1)
-   err: ErrorEstimation object
        iter: number of iterations
        E: upper bounds for teh absolute errors in matrix H
-   xb, xc: evaluation solution and error bound
-   X: the problem vector or matrix
-   H: the iteration working matrix, initially X'H = 0 and <X,H> = R^n
-   B: Integer array of potential solutions, B[:,result.col] is a solution

The input criteria are transformed to object `cr` by method `make_criteria(criteria, x)`.
"""
function sird(X::AbstractVecOrMat{T}; γ::Real=2.0, criteria...) where T<:Real
    n, t = size(X, 1), size(X, 2)
    nt = n - t
    n > 0 && t > 0 && nt > 0 || throw(ArgumentError("no appropriate size of X: $(size(X))"))
    3γ^2 > 4 || throw(ArgumentError("γ must be > 2/√3"))

    criteria = make_criteria(criteria, X)
    H = make_H(X)
    B = Matrix{integertype(T)}(I, n, n)
    D = UnitLowerTriangular(similar(B)) # working area
    err = ErrorEstimation(H)
    while true
        err.iter += 1
        sird_step!(γ, H, B, D, err)
        result = convergence(X, H, B, err, criteria)
        result.code != 0 && return result
    end
end

function sird(X::AbstractVecOrMat{T}; γ::Real=2.0, criteria...) where T<:Complex{<:Real}
    n, t = size(X,1), size(X,2)
    XX = Matrix{real(T)}(undef, n, 2*t)
    for i in axes(X, 2)
        XX[:,i*2-1] .= real(X[:,i])
        XX[:,i*2] .= imag(X[:,i])
    end
    sird(XX; γ, criteria...)
end

function pslq(X::AbstractVector{T}; criteria...) where T<:Union{Real,Complex}
    sird(X; criteria...)
end

"""
    check_criteria(err::ErrorEstimation, x, y, criteria)

Convergence criteria - see `convergence`.
"""
function convergence_criteria(err::ErrorEstimation, criteria)
    if false
        1
    elseif condition_small_diag(err, criteria)
        2
    elseif condition_normlimit(err, criteria)
        3
    elseif err.iter > criteria.itermax
        4
    else
        0
    end
end

"""
    result = convergence(X, H, B, iter, criteria)

Termination check and provide result for `sird`.
The first `X` is the unmodified problem matrix.
The arguments `H`, `B` are as described in the algoritm paper.
A solution is found, if `X'B` has a zero column.

Criteria contains the fields:
itermax
atol
rtol
elimit

result codes:
    0: means that iterations should continue - no convergence yet achieved
    1: convergence has been reached (evaluation of X'B)
    2: H[n-t,n-t] is (numerically) zero, but now solution column in `B` found.
       ( maybe retry with higher precision)
    3: There is no solution with euclidean norm <= limit
    4: Iteration maximum reached - no solution found
"""
function convergence(X::AbstractVecOrMat{T}, H::AbstractMatrix, B::AbstractMatrix,
    err::ErrorEstimation, criteria::NamedTuple) where T

    col = check_evaluation(X, B, criteria)
    code = col > 0 ? 1 : convergence_criteria(err, criteria)

    (;code, col, err, X, B, H)
end

function check_evaluation(X::AbstractVecOrMat{T}, B::AbstractMatrix, criteria) where T
    n, t = size(X, 1), size(X, 2)
    nt = n - t
    function xtb2(j, i)
        s1 = T(0)
        s2 = real(T)(0)
        for k = 1:n
            bkjx = B[k, j] * X[k, i]
            s1 += bkjx
            s2 += abs(bkjx)
        end
        abs(s1), s2
    end
    function xtb(j)
        errmax, t1, t2 = zeros(Float64, 3)
        for i = 1:t
            s1, s2 = xtb2(j, i)
            if s1 > errmax * s2
                errmax = s1 / s2
                t1, t2 = s1, s2
            end
        end
        t1, t2
    end
    for j = 1:n
        s1, s2 = xtb(j)
        if condition_relabs(s1, s2, criteria)
            return j
        end
    end
    return 0
end

function make_criteria(criteria, x)
    return (rtol=10*eps(float(eltype(x))), atol=10*eps(norm(x)), itermax=10000, elimit=1e10, γ=2.0)
end

function condition_relabs(x, y, criteria)
    abs(x) <= y * criteria.rtol + criteria.atol
end

function condition_small_diag(err, criteria)
    H = err.H
    E = err.E
    nt = size(H, 2)
    condition_relabs(H[nt, nt], E[nt, nt], criteria)
end

function condition_normlimit(err, criteria)
    maximum(abs, diag(err.H)) <= inv(criteria.elimit)
end
