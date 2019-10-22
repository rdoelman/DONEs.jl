module DONEs

using Distributions
using LinearAlgebra
using NLopt

include("src\\RFEs.jl")

mutable struct DONE
    rfe::RFE # Random Feature Expansion

    # Variables
    current_optimal_x::Vector{T} where T <: AbstractFloat
    n::Int # number of variables
    lower_bound::Vector{T} where T <: AbstractFloat
    upper_bound::Vector{T} where T <: AbstractFloat

    # DONE algorithm sliding window variant
    sliding_window::Bool
    sliding_window_length::Int
    past_inputs::Vector{Vector{T}} where T <: AbstractFloat
    past_outputs::Vector{T} where T <: AbstractFloat

    # Algorithm variables
    iteration::Int

    # Exploration
    surrogate_exploration_prob_dist::Distributions.Distribution
    function_exploration_prob_dist::Distributions.Distribution
end

function DONE(rfe, lower_bound, upper_bound, σ_surrogate_exploration, σ_function_exploration; sliding_window=true, sliding_window_length=1 )
    n = size(rfe.Ω,2)
    current_optimal_x = (lower_bound+upper_bound)./2.0
    if sliding_window
        past_inputs = Vector{Float64}[]
        past_outputs = Float64[]
    else
        sliding_window_length=0
        past_inputs = Vector{Float64}[]
        past_outputs = Float64[]
    end
    iteration = 0
    surrogate_exploration_prob_dist = Distributions.MvNormal(zeros(Float64,n),σ_surrogate_exploration*Diagonal(ones(Float64,n)))
    function_exploration_prob_dist = Distributions.MvNormal(zeros(Float64,n),σ_function_exploration*Diagonal(ones(Float64,n)))

    DONE(rfe,current_optimal_x,n,lower_bound,upper_bound,sliding_window,
        sliding_window_length,past_inputs,past_outputs,iteration,
        surrogate_exploration_prob_dist,function_exploration_prob_dist)
end

# page 111 of Laurens Bliek - Automatic Tuning of Photonic Beamformers
function add_measurement!(alg::DONE,x::Vector{T} where T <: AbstractFloat,y::AbstractFloat)
    v = alg.rfe.variable_offset ? alg.rfe.offset : 0.

    # downdate with oldest measurement
    if alg.sliding_window && alg.sliding_window_length + 1 <= alg.iteration
        a, g = downdateRFE!(alg.rfe, alg.past_inputs[1], alg.past_outputs[1]+v)
        if alg.rfe.variable_offset
            alg.rfe.h[:] = alg.rfe.h - g*(1.0 - dot(a,alg.rfe.h))
        end

        # update list of inputs and measurements
        alg.past_inputs[:] = vcat(alg.past_inputs[2:end],[x])
        alg.past_outputs[:] = vcat(alg.past_outputs[2:end],y)
    elseif alg.sliding_window && alg.sliding_window_length + 1 > alg.iteration
        alg.past_inputs[:] = push!(alg.past_inputs,x)
        alg.past_outputs[:] = push!(alg.past_outputs,y)
    end

    # account for variable offset if any
    if alg.rfe.variable_offset && y + v > 0
        alg.rfe.offset = -2y
        alg.rfe.c[:] = alg.rfe.c + (-2y - v)*alg.rfe.h
        v = -2y
    end

    # update with newest measurement
    a,g = updateRFE!(alg.rfe,x,y+v)

    # account for variable offset if any
    if alg.rfe.variable_offset
        alg.rfe.h[:] = alg.rfe.h + g*(1.0 - dot(a,alg.rfe.h))
    end

    alg.iteration = alg.iteration + 1
end

function project_on_bounds(x,lb,ub)
    return [min(max(xi,lbi),ubi) for (xi,lbi,ubi) in zip(x,lb,ub)]
end

function update_optimal_input!(alg::DONE)
    Ω = alg.rfe.Ω
    b = alg.rfe.b
    c = alg.rfe.c
    o = alg.rfe.offset

    f(x) = dot(c, cos.(Ω*x + b)) + o
    ∇f(x) = -Ω' * Diagonal(sin.(Ω*x + b)) * c
    function myfunc(x::Vector,grad::Vector)
        grad[:] = ∇f(x)
        return f(x)
    end

    opt = NLopt.Opt(:LD_LBFGS,alg.n)
    opt.lower_bounds = alg.lower_bound
    opt.upper_bounds = alg.upper_bound
    opt.min_objective = myfunc

    x0 = project_on_bounds(alg.current_optimal_x + rand(alg.surrogate_exploration_prob_dist),alg.lower_bound,alg.upper_bound)
    (minf,minx,ret) = NLopt.optimize(opt, zeros(Float64,alg.n))
    alg.current_optimal_x[:] = minx
    return minx
end

function new_input(alg::DONE)
    return project_on_bounds(alg.current_optimal_x + rand(alg.function_exploration_prob_dist),alg.lower_bound,alg.upper_bound)
end

end # module
