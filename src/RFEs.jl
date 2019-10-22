# Random Feature/Fourier Expansion

"""
    RFE(D::Int, # number of basis functions
        n::Int, # number of input variables

        Ω::Matrix{T} where T <: AbstractFloat, # random feature coefficients
        b::Vector{T} where T <: AbstractFloat, # random feature phase

        c::Vector{T} where T <: AbstractFloat, # basis function coefficients
        P::Matrix{T} where T <: AbstractFloat, # P is the covariance matrix of the coefficients
        λ::AbstractFloat, # regularization parameter of the coefficients

        variable_offset::Bool,
        h::Vector{T} where T <: AbstractFloat, # history variable for computing offsets
        offset::AbstractFloat) # the currently used offset

A Random Feature/Fourier Expansion (RFE) is the approximation of a function y = f(x) using basis functions that are a random nonlinear mapping of the input: f(x) ≈ ∑ci gi(x).
The vector c contains the coefficients. The value of the basis functions are computed through g(x) = cos.(Ω*x + b), where Ω is of size D x n and Ω and b consist of randomly generated coefficients.
c (with inverse covariance matrix P) can be estimated from data through solving a simple linear least squares problem, typically estimated with regularization term λ||c||^2.

An offset is being calculated if variable_offset=true, so that f(x) + v = ∑(ci + hi) gi(x).
"""
mutable struct RFE
    D::Int # number of basis functions
    n::Int # number of input variables

    Ω::Matrix{T} where T <: AbstractFloat # random feature coefficients
    b::Vector{T} where T <: AbstractFloat # random feature phase

    c::Vector{T} where T <: AbstractFloat # basis function coefficients
    P::Matrix{T} where T <: AbstractFloat # P is the covariance matrix of the coefficients
    λ::AbstractFloat # regularization parameter of the coefficients

    variable_offset::Bool
    h::Vector{T} where T <: AbstractFloat # history variable for computing offsets
    offset::AbstractFloat
end

"""
    RFE(D::Int, n::Int, distribution_Ω::Distributions.Distribution, distribution_b::Distributions.Distribution, λ::AbstractFloat; variable_offset::Bool)

Pick the random coefficients of the matrix Ω and vector b from the distributions distribution_Ω and distribution_b.
"""
function RFE(D::Int, n::Int, distribution_Ω::Distributions.Distribution, distribution_b::Distributions.Distribution, λ::AbstractFloat; variable_offset::Bool)
    Ω = rand(distribution_Ω, D, n)
    b = rand(distribution_b, D)
    c = zeros(Float64,D)
    P = Diagonal(ones(Float64,D)./λ) |> Matrix

    RFE(D,n,Ω,b,c,P,λ,variable_offset,zeros(Float64,D),0.)
end

"""
    RFE(D::Int,n::Int,σ_coef::AbstractFloat;λ::AbstractFloat=1E-6,variable_offset::Bool=false)

Pick the random coefficients of the matrix Ω from N(0,σI) and b from U(0,2π)
"""
function RFE(D::Int,n::Int,σ_coef::AbstractFloat;λ::AbstractFloat=1E-6,variable_offset::Bool=false)
    distribution_Ω = Distributions.Normal(0.,σ_coef)
    distribution_b = Distributions.Uniform(0.,2π)

    RFE(D,n,distribution_Ω, distribution_b, λ, variable_offset=variable_offset)
end

"""
    featureexpand(rfe::RFE,x::Vector{T} where T <: AbstractFloat)

Compute g(x)::Vector
"""
function featureexpand(rfe::RFE,x::Vector{T} where T <: AbstractFloat)
    return cos.(rfe.Ω*x + rfe.b)
end

"""
    evaluateRFE(rfe::RFE,x::AbstractVector)

Compute ∑ci gi(x)
"""
function evaluateRFE(rfe::RFE,x::AbstractVector)
    return dot(rfe.c, featureexpand(rfe,x))
end

"""
    updateRFE!(rfe::RFE,x::Vector{T} where T <: AbstractFloat,y::AbstractFloat)

The RFE is updated in a recursive manner so that adding new measurements is easy.
The vector x is the input, and the vector y is the measured output.

See: page 37 of the dissertation of Laurens Bliek - Automatic Tuning of Photonic Beamformers
"""
function updateRFE!(rfe::RFE,x::Vector{T} where T <: AbstractFloat,y::AbstractFloat)
    an = featureexpand(rfe, x)
    γn = 1.0 / (1.0 + dot(an,rfe.P*an))
    gn = γn * rfe.P * an
    rfe.c[:] = rfe.c + gn*(y - dot(an,rfe.c)) # update coeffients
    rfe.P[:] = rfe.P - (gn * gn') ./ γn
    return an, gn
end

"""
    downdateRFE!(rfe::RFE,x::Vector{T} where T <: AbstractFloat,y::AbstractFloat)

The RFE is downdated in a recursive manner so we can remove the effect of past measurements if we use a sliding window.
The vector x is the input to be removed, and the vector y is the measured output to be removed.

See: page 111 of the dissertation of Laurens Bliek - Automatic Tuning of Photonic Beamformers
"""
function downdateRFE!(rfe::RFE,x::AbstractVector{Float64},y::AbstractFloat)
    an = featureexpand(rfe, x)
    γn = 1.0 / (1.0 - dot(an,rfe.P*an))
    gn = γn * rfe.P * an
    rfe.c[:] = rfe.c - gn*(y - dot(an,rfe.c)) # update coeffients
    rfe.P[:] = rfe.P + (gn * gn') ./ γn
    return an, gn
end
