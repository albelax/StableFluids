#ifndef _RAND_GPU_H
#define _RAND_GPU_H

#include <vector>

namespace Rand_GPU
{
    /// Fill up a vector on the device with n floats. Memory is arrumed to have been preallocated.
    int randFloatsInternal(float *&/*devData*/, const size_t /*n*/);

    /// Given an stl vector of floats, fill it up with random numbers
    int randFloats(std::vector<float>&);
    int randFloats(float *&devData, const size_t n);
    void sort();
}

class Pippo
{
public:
    Pippo() = default;
    void print();
private:
    unsigned int N = 5000;
    unsigned int maxThreadsPerBlock = 1024;
    unsigned int numBlocks = N / maxThreadsPerBlock + 1;
    float *host_A;
    float *dev_A;
    float *dev_B;
    float *dev_C;
};

#endif //_RAND_GPU_H
