#ifndef _GPU_SOLVER_KERNELS_H
#define _GPU_SOLVER_KERNELS_H

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

// Cuda includes begin
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include "tuple.h"

// constant memory is a small chunk of memory off chip,
// slower than registers and L1 cache, but a lot faster than global memory
// since these values will be used often it's worth storing them
// in constant memory instead of feeding them to the gpu when the kernel is launched
extern __constant__ unsigned int c_gridSize[2];
extern __constant__ unsigned int c_rowVelocity[2];
extern __constant__ unsigned int c_columnVelocity[2];
extern __constant__ unsigned int c_totVelocity[2];


//----------------------------------------------------------------------------------------------------------------------

__global__ void d_vectorAdd( real *sum, real *A, real *B, size_t arrayLength );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setPvx(tuple<real> * _pvx);


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setPvy( tuple<real> * _pvy );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_reset( real * _in, unsigned int arrayLength );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setVelBoundaryX( real * _velocity );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setVelBoundaryY( real * _velocity );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_setCellBoundary( real *_value );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_gather( real * _value, unsigned int _size );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_projection( real * _pressure, real * _divergence );

//----------------------------------------------------------------------------------------------------------------------


__global__ void d_divergenceStep( real * _pressure, real * _divergence, tuple<real *> _velocity );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_velocityStep( real * _pressure, tuple<real *> _velocity );


//----------------------------------------------------------------------------------------------------------------------


__global__ void d_advectVelocity(tuple<real *> _previousVelocity, tuple<real *> _velocity,
                                           tuple<real> * _pvx, tuple<real> * _pvy, real _timestep );


//----------------------------------------------------------------------------------------------------------------------

__global__ void d_advectCell( real * _value, real * _value0, tuple<real *> _velocity, real _timestep );

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_diffuseVelocity( tuple<real *> _previousVelocity, tuple<real *> _velocity, real _timestep, real _diffusion );

//----------------------------------------------------------------------------------------------------------------------

__device__ real velocityBoundary(tuple<real *> _velocity , int _idx, int _idy);

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_diffuseCell(real * _previousDensity, real * _density, real _timestep, real _viscosity );

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_addVelocity_x( real * _previousVelocity, real * _velocity );

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_addVelocity_y( real * _previousVelocity, real * _velocity );

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_addDensity( real * _previousDensity, real * _density );

//----------------------------------------------------------------------------------------------------------------------
#endif
