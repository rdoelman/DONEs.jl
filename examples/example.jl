using DONEs
using Random
using LinearAlgebra

Random.seed!(1)

n = 3
rfe = RFE(20,n,0.1)

lb = -ones(n)
ub = ones(n)
σs = 0.1 # surrogate exploration
σf = 0.1 # function exploration
done = DONE(rfe, lb, ub, σs, σf)

f(x) = dot(x.-0.1,x.-0.1) + 0.05*randn() # simple quadratic function
N = 40
for i in 1:N
    xi = new_input(done)
    yi = f(xi)
    add_measurement!(done, xi, yi)
    update_optimal_input!(done)
end

println("Estimated minimum:")
evaluateRFE(rfe, done.current_optimal_x) |> display
println("Estimated minimizer")
done.current_optimal_x |> display
