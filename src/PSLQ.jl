module PSLQ
using LinearAlgebra

export pslq, sird

@enum ReturnCode CONTINUE=0 SUCCESS=1 INEXACT=2 LIMITED=3 ITMAX=4

mutable struct ErrorEstimation{MH<:AbstractMatrix,ME<:AbstractMatrix}
    iter::Int
    H::MH
    E::ME
    function ErrorEstimation(H::MH) where MH<:AbstractMatrix
        E = (Float64 ∘ abs).(H)
        ME = typeof(E)
        new{MH,ME}(0, H, E)
    end
end

"""
    increase_iter(err::ErrorEstimation)

Increase step counter and scale res.E 1/2^res.iter
"""
function increase_iter(err::ErrorEstimation)
    err.iter += 1
    err.E ./= 2.0
    nothing
end

function clean_H!(err::ErrorEstimation, i::Integer, k::Integer)
    H = err.H
    hik = H[i, k]
    iszero(hik) && return hik
    eik = ldexp(err.E[i, k], err.iter)*eps(typeof(real(hik)))
    hikold = hik
    if abs(hik) <= eik
        H[i, k] = hik = zero(hik)
    elseif eltype(H) <: Complex
        hr, hi = reim(hik)
        if abs(hr) <= eik
            H[i,k] = hik = Complex(zero(hr), hi)
        elseif abs(hi) <= eik
            H[i,k] = hik = Complex(hr)
        end
    end
    if hik != hikold
    print("clean_H($i,$k) h=$(ComplexF64(hikold)), e=$eik")
    println(", => $(ComplexF64(hik))")
    end
    hik
end

function Base.show(io::IO, err::ErrorEstimation)
    println(io, "ErrorEstimation(iter=", err.iter, ")")
end

struct SirdResult{TE<:ErrorEstimation,TB,TX,NT}
    code::ReturnCode
    col::Int
    err::TE
    criteria::NT
    B::TB
    X::TX
    function SirdResult(code::ReturnCode, col::Integer, err::TE, crit::NT, B::TB, X::TX) where {TE,TB,TX,NT}
        new{TE,TB,TX,NT}(code, col, err, crit, B, X)
    end
end
function Base.propertynames(sr::SirdResult)
    fsr = fieldnames(typeof(sr))
    ferr = propertynames(sr.err)
    fcr = propertynames(sr.criteria)
    (fsr..., ferr..., fcr..., :solution, :limit)
end

function Base.getproperty(sr::S, s::Symbol) where {TE,S<:SirdResult{TE}}
    err = getfield(sr, :err)
    cr = getfield(sr, :criteria)
    if s in fieldnames(typeof(err))
        getproperty(err, s)
    elseif s in fieldnames(typeof(cr))
        getproperty(cr, s)
    elseif s === :solution
        solution_column(sr)
    elseif s === :limit
        Float64(1/maximum(abs, diag(err.H)))
    else
        getfield(sr, s)
    end
end

function Base.show(io::IO, sr::SirdResult)
    println(io, "SirdResult(code=", sr.code, ", colum=", sr.col, ", iter=", sr.iter, ")")
    if sr.code == 1
        println(io, sr.B[:, sr.col])
    end
end


#include("pslq.jl")
#include("iterations.jl")
include("sird.jl")
include("sird_iterations.jl")

end # PSLQ
