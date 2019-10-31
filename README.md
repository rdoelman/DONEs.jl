# DONEs.jl

## Introduction
DONE (Data-based Online Non-linear Extremum-seeker) is an algorithm for finding the minimum of an unknown function that is typically expensive or difficult to evaluate and that returns a 'measurement' that is (perhaps) corrupted by noise.
Typical examples of such functions are ones that require a costly experiment, physical measurements or a long simulation.
The purpose of the underlying algorithm is to find the minimum of this unknown function in as few measurements as possible.

For details of the underlying algorithm, see:
- L. Bliek, H.R.G.W. Verstraete, M. Verhaegen and S. Wahls - [Online otimization with costly and noisy measurements using random Fourier expansions](https://arxiv.org/abs/1603.09620).
- L. Bliek - [Automatic Tuning of Photonic Beamformers](https://repository.tudelft.nl/islandora/object/uuid:8bf73354-7c68-4512-8c2b-a5f060e783f4/datastream/OBJ/download).

The original creators of the algorithm have written implementations in:
- [C++](https://bitbucket.org/csi-dcsc/donecpp/src/master/)
- [Python](https://bitbucket.org/csi-dcsc/pydonec/src/master/)
- [Matlab](https://bitbucket.org/csi-dcsc/done_matlab/src/master/)

This implementation is in Julia, using NLopt.jl for optimization.

## Usage
The algorithm keeps track of a surrogate function g(x), an estimate of the real, unknown function f(x), based on previous measurements.
The surrogate function is a Random Fourier (/Feature) Expansion (RFE).

To create a surrogate function with length(x) = 3, using 20 basis functions, and using a standard deviation of 0.1 in the normal distribution used for generating random features, use
```julia
n = 3
rfe = RFE(20,n,0.1)
```
The number of basis functions and the standard deviation used above are instrumental for how well g(x) can approximate f(x). These need to be tuned by the user.

The DONE algorithm expects x to have an element-wise lower and upper bound.
It uses two exploration parameters:
- one to initialize an optimizer (NLopt's L-BFGS) away from the current estimate of the optimal x,
- ones to explore new measurements away from the current estimate of the optimal x.

For example:
```julia
lb = -ones(n) # lower bound
ub = ones(n) # upper bound
σs = 0.1 # surrogate exploration  
σf = 0.1 # function exploration
done = DONE(rfe, lb, ub, σs, σf)
```
If the function f(x) changes (slowly) over time, you can use a sliding window of measurements. See the function documentation of DONE to see how the keywords need to be set.

The idea is to ask DONE for a new x to test, do the measurement, give this to DONE, estimate where the optimum is and repeat.
```julia
f(x) = dot(x.-0.1,x.-0.1) + 0.05*randn() # simple quadratic function
N = 40
for i in 1:N
    xi = new_input(done)
    yi = f(xi)
    add_measurement!(done, xi, yi)
    update_optimal_input!(done)
end
```
