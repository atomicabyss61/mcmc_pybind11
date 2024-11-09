import sys
sys.path.append('C:/Users/Abhi/Desktop/Code/mcmc_pybind11/build/debug')
import sampling_simulations
import matplotlib.pyplot as plt
import math
import time
import collections
import random

def sampling_dist(x: float) -> float:
    return math.sin(x)

def unif_pmf(x: float) -> float:
    return 1/math.pi

def unif_sampler(x: float) -> float:
    return math.pi*x

samples_per_iteration = 500
def rejection_sampling(sampling_distribution, proposed_distribution, proposed_sampler, k: int, samples: int):
    accepted_samples = []
    proposed_samples = collections.deque()
    
    while len(accepted_samples) < samples:
        for i in range(samples_per_iteration):
            proposed_samples.append(proposed_sampler(random.uniform(0, 1)))
        
        while len(accepted_samples) < samples and len(proposed_samples) != 0:
            sample = proposed_samples.popleft()
            if random.uniform(0, 1) <= sampling_distribution(sample) / (k * proposed_distribution(sample)):
                accepted_samples.append(sample)
    
    return accepted_samples

samples = 1000000
start_time_c = time.time()
result_c = sampling_simulations.rejection_sampling(sampling_dist, unif_pmf, unif_sampler, 5, samples)
end_time_c = time.time()
print(f"c++ function took {end_time_c - start_time_c} seconds to run {len(result_c)}")

start_time_p = time.time()
result_p = rejection_sampling(sampling_dist, unif_pmf, unif_sampler, 5, samples)
end_time_p = time.time()
print(f"python function took {end_time_p - start_time_p} seconds to run")

t = list(range(samples))

plt.hist(result_c, bins = 50, alpha=0.7)
plt.show()