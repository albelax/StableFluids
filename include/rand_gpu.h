#ifndef _RAND_GPU_H
#define _RAND_GPU_H

#include <vector>

#define CUDA_CALLABLE_MEMBER __host__ __device__

namespace Rand_GPU
{
    /// Fill up a vector on the device with n floats. Memory is arrumed to have been preallocated.
    int randFloatsInternal(float *&/*devData*/, const size_t /*n*/);

    /// Given an stl vector of floats, fill it up with random numbers
    int randFloats(std::vector<float>&);

    void hello();
}

class Pippo
{
public:
    Pippo() = default;
    void print();
};

#endif //_RAND_GPU_H
