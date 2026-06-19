module PSLQ
using LinearAlgebra

export pslq, sird

@enum ReturnCode CONTINUE=0 SUCCESS=1 INEXACT=2 LIMITED=3 ITMAX=4

mutable struct ErrorEstimation{MH<:AbstractMatrix,ME<:AbstractMatrix}
    iter::Int
    H::MH
    E::ME
    function ErrorEstimation(H::MH) where MH<:AbstractMatrix
        E = (abs ∘ Float64).(H)
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
        sr.code == SUCCESS ? sr.B[:, sr.col] : vec(sr.B[1:0])
    elseif s === :limit
        Float64(floor(1/maximum(abs, diag(err.H))))
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
