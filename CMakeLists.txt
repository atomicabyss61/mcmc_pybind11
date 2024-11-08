cmake_minimum_required(VERSION 3.10)
project(mcmc_pybind11)
set(PYBIND11_FINDPYTHON ON)
set(CMAKE_CXX_STANDARD 20)
find_package(pybind11 REQUIRED)

pybind11_add_module(sampling_simulations src/sampling_simulations.cpp)