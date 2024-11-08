import sys
sys.path.append('C:/Users/Abhi/Desktop/Code/mcmc_pybind11/build/debug')
import sampling_simulations

def sampling_dist(x: float) -> float:
    print("in py: sampling_dist")
    return pow(x, 3) / 3

def unif_pmf(x: float) -> float:
    print("in py: unif_pmf")
    return 1

def unif_sampler(x: float) -> float:
    print("in py: unif_sampler")
    return x

result = sampling_simulations.rejection_sampling(sampling_dist, unif_pmf, unif_sampler, 1, 100)
print(result)