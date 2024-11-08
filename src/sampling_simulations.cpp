#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "sampling_simulations.h"

namespace py = pybind11;

#define SAMPLES_PER_ITERATION 100

namespace mc_simulation
{

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::mutex queue_mutex;
    std::condition_variable cond_var;
    bool finished_sampling = false;

    /***
     * Function generates samples
     */
    void generate_proposal_samples(std::queue<double> &s_queue, std::mutex &queue_mutex, std::condition_variable &cond_var,
                                   std::function<double(double)> &proposal_sample_generator)
    {
        while (!finished_sampling)
        {
            std::vector<double> generated_samples(SAMPLES_PER_ITERATION, 0.0);
            // std::cout << "in generate proposals"
            {
                pybind11::gil_scoped_acquire acquire;
                for (int i = 0; i < SAMPLES_PER_ITERATION; i++)
                {
                    double unif_sample = dis(gen);
                    double prop_sample = proposal_sample_generator(unif_sample);
                    generated_samples.push_back(prop_sample);
                }
            }

            {

                std::unique_lock<std::mutex> lock(queue_mutex);
                cond_var.wait(lock, [&s_queue]
                              { return s_queue.size() < SAMPLES_PER_ITERATION || finished_sampling; });
                for (double sample : generated_samples)
                {
                    s_queue.push(sample);
                }
                cond_var.notify_one();
            }

            // std::cout << "In generate_proposal_samples" << '\n';
            // for (int i = 0; i < SAMPLES_PER_ITERATION; i++)
            // {
            //     std::cout << "retrieving sample!" << '\n';
            //     double unif_sample = dis(gen);
            //     std::cout << "unif sample: " << unif_sample << '\n';
            //     std::cout << "checking gil state: " << PyGILState_Check() << '\n';
            //     pybind11::gil_scoped_acquire acquire;
            //     std::cout << "gil acquired" << '\n';
            //     double prop_sample = proposal_sample_generator(unif_sample);
            //     std::cout << "prop_sample: " << prop_sample << '\n';
            //     s_queue.push(prop_sample);
            //     std::cout << "sample retrieved!" << '\n';
            // }
        }
    }

    /***
     * Function checks queued samples to accept or reject
     */
    void sample_accept_reject(std::queue<double> &s_queue, std::mutex &queue_mutex, std::condition_variable &cond_var,
                              std::function<double(double)> &sample_distribution, std::function<double(double)> &proposal_distribution,
                              std::vector<double> &total_samples, int samples, int k)
    {
        for (int samples_found = 0; samples_found < samples; i++)
        {
            std::vector<double> rem_samples(SAMPLES_PER_ITERATION, 0.0);
            int added_idx = 0;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cond_var.wait(lock, [&s_queue]
                              { return s_queue.size() > SAMPLES_PER_ITERATION || finished_sampling; });

                if (finished_sampling)
                {
                    return;
                }

                for (int i = 0; i < SAMPLES_PER_ITERATION && !s_queue.empty(); i++)
                {
                    double sample = s_queue.front();
                    s_queue.pop();
                    rem_samples[i] = sample;
                    added_idx++;
                }
                cond_var.notify_one();
            }

            std::vector<double> accepted_samples(SAMPLES_PER_ITERATION, 0.0);
            int accepted_idx = 0;
            {
                pybind11::gil_scoped_acquire acquire;
                for (int i = 0; i < added_idx; i++)
                {
                    double proposed_sample = rem_samples[i];
                    if (sample_distribution(proposed_sample) / (k * proposal_distribution(proposed_sample)) <= dis(gen))
                    {
                        accepted_samples[i] = proposed_sample;
                        accepted_idx++;
                    }
                }
            }

            for (int j = 0; j < accepted_idx; j++)
            {
                total_samples[samples_found] = accepted_samples[j];
                samples_found++;
            }
        }

        finished_sampling = true;
    }

    /***
     *  Rejection sampling for univariate distributions
     */
    std::vector<double> rejection_sampling(
        std::function<double(double)> &sample_distribution,
        std::function<double(double)> &proposal_distribution,
        std::function<double(double)> &proposal_sample_generator,
        int k,
        int samples)
    {

        // generate samples from proposed distribution
        std::queue<double> sample_queue;

        // plug values into f(x)/(k*g(x)) sequentially until we have the required samples
        std::cout << "Launching threads!" << '\n';
        pybind11::initialize_interpreter();
        std::vector<double> accepted_samples(samples, 0.0);
        std::jthread t([&]()
                       { generate_proposal_samples(sample_queue, queue_mutex, cond_var, proposal_sample_generator); });
        std::jthread a([&]()
                       { sample_accept_reject(sample_queue, queue_mutex, cond_var, sample_distribution, proposal_distribution, accepted_samples, samples, k); });
        return accepted_samples;
    }

    int simple_add(int i, int j)
    {
        return i + j;
    }
};

PYBIND11_MODULE(sampling_simulations, m)
{
    m.doc() = "Module that performs sampling simulations";
    m.def("rejection_sampling", &mc_simulation::rejection_sampling, "function that performs rejection sampling");
    m.def("simple_add", &mc_simulation::simple_add, "function that is a test add");
}