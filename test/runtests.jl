using DONEs
using Test
using Distributions
using LinearAlgebra
using Random
Random.seed!(1)

@testset "DONEs.jl" begin
    # Write your own tests here.

    @testset "RFE / Random Fourier Expansion" begin

        # case n > 1
        D = 10
        b = 2π*rand(D)
        c = randn(D)
        P = zeros(Float64,D,D) + Diagonal(ones(D))

        λ = 1E-6
        variable_offset = false
        h = zeros(Float64,D)
        offset = 0.

        @testset "Construction" for n in (1,4)
            Ω = randn(D,n)
            @test RFE(D,n,Ω,b,c,P,λ,variable_offset,h,offset) isa RFE
            @test RFE(D,n,0.1) isa RFE
            @test RFE(D,n,Distributions.Normal(0.,1.),Distributions.Uniform(0.,2π),λ=λ,variable_offset=variable_offset) isa RFE
        end

        @testset "Input validation" begin
            n = 4
            Ω = randn(D,n)

            @test_throws DimensionMismatch RFE(D-1,n,Ω,b,c,P,λ,variable_offset,h,offset)
            @test_throws DimensionMismatch RFE(D,n-1,Ω,b,c,P,λ,variable_offset,h,offset)
            @test_throws DimensionMismatch RFE(D,n,randn(D-1,n),b,c,P,λ,variable_offset,h,offset)
            @test_throws DimensionMismatch RFE(D,n,Ω,randn(D-1),c,P,λ,variable_offset,h,offset)
            @test_throws DimensionMismatch RFE(D,n,Ω,b,randn(D-1),P,λ,variable_offset,h,offset)
            @test_throws DimensionMismatch RFE(D,n,Ω,b,c,randn(D-1,D),λ,variable_offset,h,offset)
            @test_throws AssertionError RFE(D,n,Ω,b,c,P,-1.,variable_offset,h,offset)
            @test_throws AssertionError RFE(D,n,Ω,b,c,Diagonal(vcat(-1.,ones(D-1))) |> Matrix,λ,variable_offset,h,offset)
            @test_throws AssertionError RFE(D,n,Ω,b,c,randn(D,D),λ,variable_offset,h,offset)
        end

        @testset "Updating and evaluating the RFE" begin
            f(x) = x^2
            n = 1
            rfe = RFE(D,n,0.1)
            P = rfe.P |> copy
            N = 40
            points = LinRange(-1,1,N)
            @test begin
                try
                    for x in points
                        DONEs.updateRFE!(rfe,[x],f(x))
                    end
                catch
                    error()
                end
                rfe.P != P
            end
            # approximation power
            @test begin
                e = [DONEs.evaluateRFE(rfe,[x]) - f(x) for x in points]
                norm(e) < 3E-2
            end
            # downdating
            @test begin
                try
                    for x in LinRange(-1,1,5)
                        DONEs.downdateRFE!(rfe,[x],f(x))
                    end
                catch
                    error()
                end
                rfe.P != P
            end
        end
    end

    @testset "DONE functions, n=$n, sliding window=$sliding_window" for n in (1,2), sliding_window in (true,false)
        D = 30
        lb = -ones(n)
        ub = ones(n)
        current_optimal_x = 0.1*rand(Float64,n)
        sliding_window_length = 100
        σ_surrogate_exploration = 0.1
        σ_function_exploration = 0.1
        f(x) = dot(x,x)
        @testset "Construction" begin
            rfe = RFE(D,n,0.1)
            @test DONE(rfe, lb, ub, σ_surrogate_exploration, σ_function_exploration,sliding_window=sliding_window, sliding_window_length=sliding_window_length ) isa DONE
        end

        # @testset "Input validation" begin
        # end

        @testset "Adding measurements" begin
            rfe = RFE(D,n,0.1)
            done = DONE(rfe, lb, ub, σ_surrogate_exploration, σ_function_exploration,sliding_window=sliding_window, sliding_window_length=sliding_window_length )
            N = 200
            points = [2*rand(n) .- 1.0 for i in 1:N]
            P = rfe.P |> copy
            @test begin
                try
                    for x in points
                        add_measurement!(done,x,f(x))
                    end
                catch
                    error()
                end
                rfe.P != P
            end
            @test done.iteration == N

            # approximation power
            @test begin
                points = [2*rand(n) .- 1.0 for i in 1:50]
                e = [DONEs.evaluateRFE(rfe,x) - f(x) for x in points]
                norm(e) < 1E-2
            end

            @testset "Optimization with NLopt of the surrogate function" begin
                @test begin
                    try
                        update_optimal_input!(done)
                    catch
                        error()
                    end
                    norm(done.current_optimal_x - zeros(n)) < 1E-2 
                end

            end
        end

    end
end
