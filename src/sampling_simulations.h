#ifndef SAMPLING_SIMULATIONS_H
#define SAMPLING_SIMULATIONS_H
#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <random>
#include <condition_variable>
#include <functional>

namespace mc_simulations
{
    std::vector<double> rejection_sampling(
        std::function<double(double)> &sample_distribution,
        std::function<double(double)> &proposal_distribution,
        std::function<double(double)> &proposal_sample_generator,
        int k,
        int samples);

    int simple_add(int i, int j);
}

#endif