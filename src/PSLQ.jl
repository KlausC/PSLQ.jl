module PSLQ
using LinearAlgebra

export pslq, sird

mutable struct ErrorEstimation{M<:AbstractMatrix}
    iter::Int
    H::M
    E::M
    function ErrorEstimation(H::M) where M<:AbstractMatrix
        new{M}(0, H::M, abs.(H))
    end
end

#include("pslq.jl")
#include("iterations.jl")
include("sird.jl")
include("sird_iterations.jl")

end # PSLQ
