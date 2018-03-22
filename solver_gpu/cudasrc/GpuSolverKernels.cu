#include "GpuSolverKernels.cuh"
#include "GpuSolver.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>


//----------------------------------------------------------------------------------------------------------------------
// KERNELS -------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

__global__ void d_setPvx( tuple<float> * _pvx, unsigned int _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if ( idx < _size )
  {
    int i = idy * _size + idx;
    _pvx[i].x = idx;
    _pvx[i].y = idy + 0.5f;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_setPvy( tuple<float> * _pvy, unsigned int _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if ( idx < _size )
  {
    int i = idy * _size + idx;
    _pvy[i].x = (float) idx + 0.5f;
    _pvy[i].y = idy;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_vectorAdd( float *sum, float *A, float *B, size_t arrayLength )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ( idx < arrayLength )
  {
    sum[idx] = A[idx] + B[idx];
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_reset( float * _in, unsigned int arrayLength )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ( idx < arrayLength )
  {
    _in[idx] = 0.0f;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_setVelBoundaryX( float * _velocity, tuple<unsigned int> _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if ( idx > 0 && idx < _size.x - 1 ) // rowsize
  {
    _velocity[idx] =  _velocity[idx + _size.x]; // set the top row to be the same as the second row
    _velocity[idx + _size.x * (_size.y-1)] = _velocity[idx + _size.x * (_size.y - 2)]; // set the last row to be the same as the second to last row
  }

  if ( idx > 0 && idx < _size.y - 1 ) // colsize
  {
    _velocity[idx * _size.x] = -_velocity[idx * _size.x + 1]; // set the first column on the left to be the same as the next
    _velocity[idx * _size.x + ( _size.x - 1)] = -_velocity[idx * _size.x + (_size.x - 2)]; // set the first column on the right to be the same as the previous
  }

  __syncthreads();

  if ( idx == 0 )
  {
    // calculating the corners
    // horrible, wasteful way of doing it
    // but for now I just need this to work

    _velocity[0] = ( _velocity[1] + _velocity[_size.x] ) / 2;

    int dst = _size.x - 1;
    int left = _size.x - 2;
    int down = _size.x + _size.x - 1;
    _velocity[dst] = (_velocity[left] + _velocity[down])/2;

    int up = (_size.y - 1) * _size.x + 1;
    left = (_size.y - 2) * _size.x;
    dst = (_size.y - 1) * _size.x;
    _velocity[dst] = (_velocity[up] + _velocity[left])/2;

    dst = (_size.y - 1) * _size.x + (_size.x -1);
    left = (_size.y - 1) * _size.x + (_size.x - 2);
    up = (_size.y - 2) * _size.x + (_size.x - 1);

    _velocity[dst] = ( _velocity[left] + _velocity[up] ) / 2;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_setVelBoundaryY( float * _velocity, tuple<unsigned int> _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if ( idx > 0 && idx < _size.x - 1 ) // rowsize
  {
    _velocity[idx] = - _velocity[idx + _size.x]; // set the top row to be the same as the second row
    _velocity[idx + _size.x * (_size.y-1)] = -_velocity[idx + _size.x * (_size.y - 2)]; // set the last row to be the same as the second to last row

  }

  if ( idx > 0 && idx < _size.y - 1 ) // colsize
  {
    _velocity[idx * _size.x] = _velocity[idx * _size.x + 1]; // set the first column on the left to be the same as the next
    _velocity[idx * _size.x + ( _size.x - 1)] = _velocity[idx * _size.x + (_size.x - 2)]; // set the first column on the right to be the same as the previous

  }

  __syncthreads();

  if ( idx == 0 )
  {
    // calculating the corners
    // horrible, wasteful way of doing it
    // but for now I just need this to work

    _velocity[0] = ( _velocity[1] + _velocity[_size.x] ) / 2;

    int dst = _size.x - 1;
    int left = _size.x - 2;
    int down = _size.x + _size.x - 1;
    _velocity[dst] = (_velocity[left] + _velocity[down])/2;

    int up = (_size.y - 1) * _size.x + 1;
    left = (_size.y - 2) * _size.x;
    dst = (_size.y - 1) * _size.x;
    _velocity[dst] = (_velocity[up] + _velocity[left])/2;

    dst = (_size.y - 1) * _size.x + (_size.x -1);
    left = (_size.y - 1) * _size.x + (_size.x - 2);
    up = (_size.y - 2) * _size.x + (_size.x - 1);

    _velocity[dst] = ( _velocity[left] + _velocity[up] ) / 2;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_setCellBoundary( float * _value , tuple<unsigned int> _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if ( idx > 0 && idx < _size.x - 1 )
  {
    _value[idx] = _value[idx + _size.x];
    _value[idx + _size.x * (_size.y-1)] = _value[idx + _size.x * (_size.y - 2)];
  }

  if ( idx > 0 && idx < _size.y - 1 )
  {
    _value[idx * _size.x] = _value[idx * _size.x + 1]; // set the first column on the left to be the same as the next
    _value[idx * _size.x + ( _size.x - 1)] = _value[idx * _size.x + (_size.x - 2)];
  }
  __syncthreads();

  if ( idx == 0 )
  {
    // again
    // calculating the corners
    // horrible, wasteful way of doing it
    // but for now I just need this to work

    _value[0] = ( _value[1] + _value[_size.x] ) / 2;

    int dst = _size.x - 1;
    int left = _size.x - 2;
    int down = _size.x + _size.x - 1;
    _value[dst] = (_value[left] + _value[down]) / 2;

    int up = (_size.y - 1) * _size.x + 1;
    left = (_size.y - 2) * _size.x;
    dst = (_size.y - 1) * _size.x;
    _value[dst] = (_value[up] + _value[left])/2;

    dst = (_size.y - 1) * _size.x + (_size.x -1);
    left = (_size.y - 1) * _size.x + (_size.x - 2);
    up = (_size.y - 2) * _size.x + (_size.x - 1);

    _value[dst] = ( _value[left] + _value[up] ) / 2;
  }

}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_gather( float * _value, unsigned int _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__ float localValue[];

  if ( idx > 0 && idx < _size - 1 )
  {
    //    printf(" idx: %d, idy: %d\n", idx, idy );
    localValue[idx] = ( _value[idx - 1] + _value[idx] + _value[idx + 1] );
    __syncthreads();
    _value[idx] = localValue[idx];

  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_gather2D( float * _value, unsigned int _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  extern __shared__ float localValue[];

  if ( idx < _size )
  {
    //    printf(" idx: %d, idy: %d\n", idx, idy );
    localValue[idx] = idx - idy; //( _value[idx - 1] + _value[idx] + _value[idx + 1] );
    __syncthreads();
    _value[idx] = localValue[idx];
  }
}



//----------------------------------------------------------------------------------------------------------------------

__global__ void d_projection( float * _pressure, float * _divergence, tuple<float *> _velocity,
                              tuple<unsigned int> _rowVelocity, tuple<unsigned int> _columnVelocity,
                              tuple<unsigned int> _gridSize)
{
  extern __shared__ float local_divergence[];

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if ( idx > 0 && idx < _gridSize.x - 1 &&
       idy > 0 && idy < _gridSize.y - 1 )
  {
    int currentCell = idy * _gridSize.x + idx;

    int right = idy * _rowVelocity.x + (idx + 1);
    int currentVelX = idy * _rowVelocity.x + idx;
    int down = (idy + 1) * _rowVelocity.y + idx;
    int currentVelY = idy * _rowVelocity.y + idx;

    int sIdx = threadIdx.y * 9 + threadIdx.x;
    local_divergence[sIdx] = _velocity.x[right] - _velocity.x[currentVelX] + _velocity.y[down] - _velocity.y[currentVelY];

    _pressure[currentCell] = 0;
    _divergence[currentCell] = local_divergence[sIdx];
  }
}

//----------------------------------------------------------------------------------------------------------------------
