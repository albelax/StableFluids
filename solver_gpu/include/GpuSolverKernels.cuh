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
#include "tuple.h"

__global__ void d_vectorAdd( real *sum, real *A, real *B, size_t arrayLength );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setPvx(tuple<real> * _pvx, unsigned int _size );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setPvy( tuple<real> * _pvy, unsigned int _size );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_reset( real * _in, unsigned int arrayLength );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setVelBoundaryX( real * _velocity, tuple<unsigned int> _size );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setVelBoundaryY( real * _velocity, tuple<unsigned int> _size );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setCellBoundary(real *_value, tuple<unsigned int> _size );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_gather( real * _value, unsigned int _size );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_projection(real * _pressure, real * _divergence,
                             tuple<unsigned int> _gridSize);

//----------------------------------------------------------------------------------------------------------------------


__global__ void d_divergenceStep(real * _pressure, real * _divergence, tuple<real *> _velocity,
                                 tuple<unsigned int> _rowVelocity,
                                 tuple<unsigned int> _gridSize);


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_velocityStep(real * _pressure, tuple<real *> _velocity,
                               tuple<unsigned int> _rowVelocity, tuple<unsigned int> _columnVelocity,
                               tuple<unsigned int> _gridSize);


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_advectVelocity(tuple<real *> _previousVelocity, tuple<real *> _velocity,
                                           tuple<real> * _pvx, tuple<real> * _pvy,
                                           tuple<unsigned int> _rowVelocity,
                                           tuple<unsigned int> _columnVelocity,
                                           tuple<unsigned int> _gridSize );


//----------------------------------------------------------------------------------------------------------------------
