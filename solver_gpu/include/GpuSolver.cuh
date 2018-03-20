// Cuda includes begin
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
// cuda includes end
#include "GpuSolver.h"

__global__ void vectorAdd( float *sum, float *A, float *B, size_t arrayLength );

__global__ void setPvx(tuple<float> * _pvx, unsigned int _size );

__global__ void setPvy( tuple<float> * _pvy, unsigned int _size );

__global__ void d_reset( float * _in, unsigned int arrayLength );
