#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include "sampling_simulations.h"

namespace py = pybind11;

#define SAMPLES_PER_ITERATION 500

class Spinlock
{
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

public:
    void lock()
    {
        while (flag.test_and_set(std::memory_order_acquire))
        {
            // Busy-wait (spin)
        }
    }

    void unlock()
    {
        flag.clear(std::memory_order_release);
    }
};

namespace mc_simulation
{

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    // std::mutex queue_mutex;
    // std::condition_variable cond_var;
    bool finished_sampling = false;
    Spinlock spinlock;

    /***
     * Function generates samples
     */
    void generate_proposal_samples(std::queue<double> &s_queue, std::function<double(double)> &proposal_sample_generator)
    {
        while (!finished_sampling)
        {
            std::vector<double> generated_samples(SAMPLES_PER_ITERATION);
            std::generate(generated_samples.begin(), generated_samples.end(), [&]()
                          { return dis(gen); });
            {
                pybind11::gil_scoped_acquire acquire;
                for (int i = 0; i < SAMPLES_PER_ITERATION; i++)
                {
                    // double unif_sample = dis(gen);
                    // double prop_sample = proposal_sample_generator(unif_sample);
                    // generated_samples.push_back(prop_sample);
                    generated_samples[i] = proposal_sample_generator(generated_samples[i]);
                }

                pybind11::gil_scoped_release release;
            }

            {
                // std::unique_lock<std::mutex> lock(queue_mutex);
                // cond_var.wait(lock, []
                //               { return true; });

                spinlock.lock();
                for (double sample : generated_samples)
                {
                    s_queue.push(sample);
                }
                spinlock.unlock();
                // cond_var.notify_one();
            }
        }
    }

    /***
     * Function checks queued samples to accept or reject
     */
    void sample_accept_reject(std::queue<double> &s_queue, std::function<double(double)> &sample_distribution, std::function<double(double)> &proposal_distribution,
                              std::vector<double> &total_samples, int samples, int k)
    {

        while (total_samples.size() < samples)
        {
            std::vector<double> rem_samples;
            rem_samples.reserve(SAMPLES_PER_ITERATION);
            {
                // std::unique_lock<std::mutex> lock(queue_mutex);
                // cond_var.wait(lock, [&s_queue]
                //               { return s_queue.size() > SAMPLES_PER_ITERATION || finished_sampling; });
                spinlock.lock();
                if (finished_sampling)
                {
                    return;
                }

                for (int i = 0; i < SAMPLES_PER_ITERATION && !s_queue.empty(); i++)
                {
                    double sample = s_queue.front();
                    s_queue.pop();
                    rem_samples.emplace_back(sample);
                }
                spinlock.unlock();
                // cond_var.notify_one();
            }

            std::vector<double> accepted_samples(rem_samples.size(), dis(gen));
            // accepted_samples.reserve(SAMPLES_PER_ITERATION);
            {
                pybind11::gil_scoped_acquire acquire;
                for (int s_idx = 0; s_idx < rem_samples.size(); s_idx++)
                // for (double sample : rem_samples)
                {
                    double sample = rem_samples[s_idx];
                    if (sample_distribution(sample) / (k * proposal_distribution(sample)) >= accepted_samples[s_idx])
                    {
                        accepted_samples[s_idx] = sample;
                    }
                    else
                    {
                        accepted_samples[s_idx] = -1;
                    }
                }
                // pybind11::gil_scoped_release release;
            }
            for (double acc_sample : accepted_samples)
            {
                if (acc_sample != -1)
                {
                    total_samples.emplace_back(acc_sample);
                }
                else
                {
                    continue;
                }
                if (total_samples.size() >= samples - 1)
                {
                    break;
                }
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
        std::vector<double> accepted_samples;
        accepted_samples.reserve(samples);

        // main thread acquiring gil
        pybind11::gil_scoped_release release;
        std::thread producer(generate_proposal_samples, std::ref(sample_queue), std::ref(proposal_sample_generator));
        std::thread consumer(sample_accept_reject, std::ref(sample_queue), std::ref(sample_distribution), std::ref(proposal_distribution), std::ref(accepted_samples), samples, k);

        producer.join();
        consumer.join();
        // std::cout << "sample " << accepted_samples.size() << '\n';
        return accepted_samples;
    }

};

PYBIND11_MODULE(sampling_simulations, m)
{
    m.doc() = "Module that performs sampling simulations";
    m.def("rejection_sampling", &mc_simulation::rejection_sampling, "function that performs rejection sampling");
}