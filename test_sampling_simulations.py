import sys
sys.path.append('C:/Users/Abhi/Desktop/Code/mcmc_pybind11/build/debug')
import sampling_simulations
import matplotlib.pyplot as plt
import math

def sampling_dist(x: float) -> float:
    return math.sin(x)

def unif_pmf(x: float) -> float:
    return 1/math.pi

def unif_sampler(x: float) -> float:
    return math.pi*x

samples = 1000000
result = sampling_simulations.rejection_sampling(sampling_dist, unif_pmf, unif_sampler, 5, samples)
t = list(range(samples))
print("in python printing")
plt.hist(result, bins = 50, alpha=0.7)
plt.show()