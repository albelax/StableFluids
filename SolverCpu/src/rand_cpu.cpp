#include "rand_cpu.h"
#include <random>
#include <algorithm>
#include <iostream>

/**
 * Generate a random vector of floats on the CPU, for the purposes of performance comparison.
 * \param tgt An input std::vector 
 * \return EXIT_SUCCESS when everything worked
 */
int Rand_CPU::randFloats(std::vector<real>& tgt)
{
    // Random number engine with a random device as seed    
    std::mt19937 mt(time(NULL));

    // Create a uniform distribution of floats (template parameter inferred from type)
    std::uniform_real_distribution<real> d(0.0f, 1.0f);

    // Use a lambda function to compute the generated random value within the uniform distribution
    std::generate(tgt.begin(), tgt.end(), [&] { return d(mt); });

    // Exit without problems (presumably)
    return EXIT_SUCCESS;
}
