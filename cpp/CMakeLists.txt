cmake_minimum_required(VERSION 3.10)
project(mcmc_pybind11)

find_package(pybind11 REQUIRED)

pybind11_add_module(sampleing_simulations src/sampling_simulations.cpp)