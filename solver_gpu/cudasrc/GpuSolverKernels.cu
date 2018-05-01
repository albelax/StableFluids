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
__constant__ unsigned int c_gridSize[2];
__constant__ unsigned int c_rowVelocity[2];
__constant__ unsigned int c_columnVelocity[2];
__constant__ unsigned int c_totVelocity[2];



__global__ void d_setPvx( tuple<real> * _pvx )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if ( idx < c_rowVelocity[0] )
  {
    int i = idy * c_rowVelocity[0] + idx;
    _pvx[i].x = idx;
    _pvx[i].y = idy + 0.5f;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_setPvy( tuple<real> * _pvy )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if ( idx < c_rowVelocity[1] )
  {
    int i = idy * c_rowVelocity[1] + idx;
    _pvy[i].x = (real) idx + 0.5f;
    _pvy[i].y = idy;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_vectorAdd( real *sum, real *A, real *B, size_t arrayLength )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ( idx < arrayLength )
  {
    sum[idx] = A[idx] + B[idx];
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_reset( real * _in, unsigned int arrayLength )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ( idx < arrayLength )
  {
    _in[idx] = 0.0f;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_setVelBoundaryX( real * _velocity )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if ( idx > 0 && idx < c_rowVelocity[0] - 1 ) // rowsize
  {
    _velocity[idx] =  _velocity[idx + c_rowVelocity[0]]; // set the top row to be the same as the second row
    _velocity[idx + c_rowVelocity[0] * (c_columnVelocity[0]-1)] = _velocity[idx + c_rowVelocity[0] * (c_columnVelocity[0] - 2)]; // set the last row to be the same as the second to last row
  }

  if ( idx > 0 && idx < c_columnVelocity[0] - 1 ) // colsize
  {
    _velocity[idx * c_rowVelocity[0]] = -_velocity[idx * c_rowVelocity[0] + 1]; // set the first column on the left to be the same as the next
    _velocity[idx * c_rowVelocity[0] + ( c_rowVelocity[0] - 1)] = -_velocity[idx * c_rowVelocity[0] + (c_rowVelocity[0] - 2)]; // set the first column on the right to be the same as the previous
  }

  __syncthreads();

  if ( idx == 0 )
  {
    // calculating the corners
    // horrible, wasteful way of doing it
    // but for now I just need this to work

    _velocity[0] = ( _velocity[1] + _velocity[c_rowVelocity[0]] ) / 2;

    int dst = c_rowVelocity[0] - 1;
    int left = c_rowVelocity[0] - 2;
    int down = c_rowVelocity[0] + c_rowVelocity[0] - 1;
    _velocity[dst] = (_velocity[left] + _velocity[down])/2;

    int up = (c_columnVelocity[0] - 1) * c_rowVelocity[0] + 1;
    left = (c_columnVelocity[0] - 2) * c_rowVelocity[0];
    dst = (c_columnVelocity[0] - 1) * c_rowVelocity[0];
    _velocity[dst] = (_velocity[up] + _velocity[left])/2;

    dst = (c_columnVelocity[0] - 1) * c_rowVelocity[0] + (c_rowVelocity[0] -1);
    left = (c_columnVelocity[0] - 1) * c_rowVelocity[0] + (c_rowVelocity[0] - 2);
    up = (c_columnVelocity[0] - 2) * c_rowVelocity[0] + (c_rowVelocity[0] - 1);

    _velocity[dst] = ( _velocity[left] + _velocity[up] ) / 2;
  }
  __syncthreads();

}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_setVelBoundaryY( real * _velocity )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if ( idx > 0 && idx < c_rowVelocity[1] - 1 ) // rowsize
  {
    _velocity[idx] = - _velocity[idx + c_rowVelocity[1]]; // set the top row to be the same as the second row
    _velocity[idx + c_rowVelocity[1] * (c_columnVelocity[1]-1)] = -_velocity[idx + c_rowVelocity[1] * (c_columnVelocity[1] - 2)]; // set the last row to be the same as the second to last row
  }

  if ( idx > 0 && idx < c_columnVelocity[1] - 1 ) // colsize
  {
    _velocity[idx * c_rowVelocity[1]] = _velocity[idx * c_rowVelocity[1] + 1]; // set the first column on the left to be the same as the next
    _velocity[idx * c_rowVelocity[1] + ( c_rowVelocity[1] - 1)] = _velocity[idx * c_rowVelocity[1] + (c_rowVelocity[1] - 2)]; // set the first column on the right to be the same as the previous
  }

  __syncthreads();

  if ( idx == 0 )
  {
    // calculating the corners
    // horrible, wasteful way of doing it
    // but for now I just need this to work

    _velocity[0] = ( _velocity[1] + _velocity[c_rowVelocity[1]] ) / 2;

    int dst = c_rowVelocity[1] - 1;
    int left = c_rowVelocity[1] - 2;
    int down = c_rowVelocity[1] + c_rowVelocity[1] - 1;
    _velocity[dst] = ( _velocity[left] + _velocity[down] )/2;

    int up = (c_columnVelocity[1] - 1) * c_rowVelocity[1] + 1;
    left = (c_columnVelocity[1] - 2) * c_rowVelocity[1];
    dst = (c_columnVelocity[1] - 1) * c_rowVelocity[1];
    _velocity[dst] = (_velocity[up] + _velocity[left])/2;

    dst = (c_columnVelocity[1] - 1) * c_rowVelocity[1] + (c_rowVelocity[1] -1);
    left = (c_columnVelocity[1] - 1) * c_rowVelocity[1] + (c_rowVelocity[1] - 2);
    up = (c_columnVelocity[1] - 2) * c_rowVelocity[1] + (c_rowVelocity[1] - 1);

    _velocity[dst] = ( _velocity[left] + _velocity[up] ) / 2;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_setCellBoundary( real * _value )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if ( idx > 0 && idx < c_gridSize[0] - 1 )
  {
    _value[idx] = _value[idx + c_gridSize[0]];
    _value[idx + c_gridSize[0] * (c_gridSize[1] - 1)] = _value[idx + c_gridSize[0] * (c_gridSize[1] - 2)];
  }

  if ( idx > 0 && idx < c_gridSize[1] - 1 )
  {
    _value[idx * c_gridSize[0]] = _value[idx * c_gridSize[0] + 1]; // set the first column on the left to be the same as the next
    _value[idx * c_gridSize[0] + ( c_gridSize[0] - 1)] = _value[idx * c_gridSize[0] + (c_gridSize[0] - 2)];
  }

  if ( idx == 0 )
  {
    // again
    // calculating the corners
    // horrible, wasteful way of doing it
    // but for now I just need this to work

    _value[0] = ( _value[1] + _value[c_gridSize[0]] ) / 2;

    int dst = c_gridSize[0] - 1;
    int left = c_gridSize[0] - 2;
    int down = c_gridSize[0] + c_gridSize[0] - 1;
    _value[dst] = (_value[left] + _value[down]) / 2;

    int up = (c_gridSize[1] - 1) * c_gridSize[0] + 1;
    left = (c_gridSize[1] - 2) * c_gridSize[0];
    dst = (c_gridSize[1] - 1) * c_gridSize[0];
    _value[dst] = (_value[up] + _value[left])/2;

    dst = (c_gridSize[1] - 1) * c_gridSize[0] + (c_gridSize[0] -1);
    left = (c_gridSize[1] - 1) * c_gridSize[0] + (c_gridSize[0] - 2);
    up = (c_gridSize[1] - 2) * c_gridSize[0] + (c_gridSize[0] - 1);

    _value[dst] = ( _value[left] + _value[up] ) / 2;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_gather( real * _value, unsigned int _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__ real localValue[];

  if ( idx > 0 && idx < _size - 1 )
  {
    localValue[idx] = ( _value[idx - 1] + _value[idx] + _value[idx + 1] );
    __syncthreads();
    _value[idx] = localValue[idx];
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_projection( real * _pressure, real * _divergence )
{
  // projection Step
  // this should be in a loop...
  extern __shared__ real local_pressure[];

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if ( idx > 0 && idx < c_gridSize[0] - 1 &&
       idy > 0 && idy < c_gridSize[1] - 1 )
  {
    int sIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int currentCell = idy * c_gridSize[0] + idx;

    int right = idy * c_gridSize[0] + (idx + 1);
    int left = idy * c_gridSize[0] + (idx - 1);
    int down = (idy + 1) * c_gridSize[0] + idx;
    int up = (idy - 1) * c_gridSize[0] + idx;

    local_pressure[sIdx] = ( _pressure[right] + _pressure[left] + _pressure[down] + _pressure[up] - _divergence[currentCell])/4.0;
    __syncthreads();

    _pressure[currentCell] = local_pressure[sIdx];
    //    __syncthreads();
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_divergenceStep(real * _pressure, real * _divergence, tuple<real *> _velocity )
{
  // memory shared within the block, I will treat this as a tiny 2D array,
  // the size is decided outside the kernel,
  // if the # of threads in a block is 9 the size will be 81 ( array[9][9] )
  extern __shared__ real local_divergence[];

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if ( idx > 0 && idx < c_gridSize[0] - 1 &&
       idy > 0 && idy < c_gridSize[1] - 1 )
  {
    int sIdx = threadIdx.y * blockDim.x + threadIdx.x;

    int currentCell = idy * c_gridSize[0] + idx;
    int right = idy * c_rowVelocity[0] + (idx + 1);
    int currentVelX = idy * c_rowVelocity[0] + idx;
    int down = (idy + 1) * c_rowVelocity[1] + idx;
    int currentVelY = idy * c_rowVelocity[1] + idx;

    // index of the shared memory
    local_divergence[sIdx] = _velocity.x[right] - _velocity.x[currentVelX] + _velocity.y[down] - _velocity.y[currentVelY];

    //    _pressure[currentCell] = 0.0;
    _divergence[currentCell] = local_divergence[sIdx];
    //    __syncthreads();
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_velocityStep(real * _pressure, tuple<real *> _velocity )
{
  extern __shared__ real local_velocity[];

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if ( idx > 0 && idx < c_rowVelocity[0] - 1 &&
       idy > 0 && idy < c_columnVelocity[0] - 1 )
  {
    int velocityIdx = idy * c_rowVelocity[0] + idx;
    int sIdx = threadIdx.y * blockDim.x + threadIdx.x;

    int cellIdx = idy * c_gridSize[0] + idx;
    int cellLeft = idy * c_gridSize[0] + (idx - 1);

    local_velocity[sIdx] = _pressure[cellIdx] - _pressure[cellLeft];
    //    __syncthreads();
    _velocity.x[velocityIdx] -= local_velocity[sIdx];
  }

  if ( idx > 0 && idx < c_rowVelocity[1] - 1 &&
       idy > 0 && idy < c_columnVelocity[1] - 1 )
  {

    int velocityIdx = idy * c_rowVelocity[1] + idx;
    int sIdx = threadIdx.y * blockDim.x + threadIdx.x;

    int cellIdx = idy * c_gridSize[0] + idx;
    int cellUp = (idy-1) * c_gridSize[0] + idx;

    local_velocity[sIdx] = _pressure[cellIdx] - _pressure[cellUp];
    _velocity.y[velocityIdx] -= local_velocity[sIdx];
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_advectVelocity(tuple<real *> _previousVelocity, tuple<real *> _velocity,
                                 tuple<real> * _pvx, tuple<real> * _pvy, real _timestep)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  unsigned short currentIdx = idy * c_rowVelocity[0] + idx;
  unsigned short currentIdy = idy * c_rowVelocity[1] + idx;

  if ( idx > 0 && idx < c_rowVelocity[0] - 1 &&
       idy > 0 && idy < c_columnVelocity[0] - 1 )
  {
    real nvx = _previousVelocity.x[currentIdx];
    real nvy = (_previousVelocity.y[idy * c_rowVelocity[1] + idx-1] +
        _previousVelocity.y[(idy + 1) * c_rowVelocity[1] + (idx - 1)] +
        _previousVelocity.y[currentIdy]+
        _previousVelocity.y[(idy + 1) * c_rowVelocity[1] + idx])/4;

    real oldX = _pvx[currentIdx].x - nvx * _timestep;
    real oldY = _pvx[currentIdx].y - nvy * _timestep;

    if(oldX < 0.5f) oldX = 0.5f;
    if(oldX > c_gridSize[0]-0.5f) oldX = c_gridSize[0]-0.5f;
    if(oldY < 1.0f) oldY = 1.0f;
    if(oldY > c_gridSize[1]-1.0f) oldY = c_gridSize[1]-1.0f;

    int i0 = (int)oldX;
    int j0 = (int)(oldY-0.5f);
    int i1 = i0+1;
    int j1 = j0+1;

    real wL = _pvx[j0 * c_rowVelocity[0] + i1].x-oldX;
    real wR = 1.0f-wL;
    real wB = _pvx[j1 * c_rowVelocity[0] + i0].y-oldY;
    real wT = 1.0f-wB;

    _velocity.x[currentIdx] = wB * (wL * _previousVelocity.x[j0 * c_rowVelocity[0] + i0] +
        wR * _previousVelocity.x[j0 * c_rowVelocity[0] + i1]) +
        wT * (wL * _previousVelocity.x[j1 * c_rowVelocity[0] + i0] +
        wR * _previousVelocity.x[j1 * c_rowVelocity[0] + i1]);
  }

  if ( idx > 0 && idx < c_rowVelocity[1] - 1 &&
       idy > 0 && idy < c_columnVelocity[1] - 1 )
  {
    real nvx = (
          _previousVelocity.x[(idy - 1) * c_rowVelocity[0] + idx]+
        _previousVelocity.x[(idy - 1) * c_rowVelocity[0] + (idx + 1)] +
        _previousVelocity.x[currentIdx]+
        _previousVelocity.x[idy * c_rowVelocity[0] + (idx + 1)]
        )/4;

    real nvy = _previousVelocity.y[currentIdy];

    real oldX = _pvy[currentIdy].x - nvx * _timestep;
    real oldY = _pvy[currentIdy].y - nvy * _timestep;

    if(oldX < 1.0f) oldX = 1.0f;
    if(oldX > c_gridSize[0]-1.0f) oldX = c_gridSize[0]-1.0f;
    if(oldY < 0.5f) oldY = 0.5f;
    if(oldY > c_gridSize[1]-0.5f) oldY = c_gridSize[1]-0.5f;

    int i0 = (int)(oldX-0.5f);
    int j0 = (int)oldY;
    int i1 = i0+1;
    int j1 = j0+1;

    real wL = _pvy[j0 * c_rowVelocity[1] + i1].x-oldX;
    real wR = 1.0f-wL;
    real wB = _pvy[j1 * c_rowVelocity[1] + i0].y-oldY;
    real wT = 1.0f-wB;

    _velocity.y[currentIdy] = wB * (wL * _previousVelocity.y[j0 * c_rowVelocity[1] + i0] +
        wR * _previousVelocity.y[j0 * c_rowVelocity[1] + i1]) +
        wT * (wL * _previousVelocity.y[j1 * c_rowVelocity[1] + i0] +
        wR * _previousVelocity.y[j1 * c_rowVelocity[1] + i1]);
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_advectCell( real * _value, real * _value0, tuple<real *> _velocity, real _timestep )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  unsigned short currentIdx = idy * c_gridSize[0] + idx;

  if ( idx > 0 && idx < c_gridSize[0] - 1 &&
       idy > 0 && idy < c_gridSize[1] - 1 )
  {
    real cvx = ( _velocity.x[idy * c_rowVelocity[0] + idx] + _velocity.x[idy * c_rowVelocity[0] + (idx + 1)] ) / 2.0f;
    real cvy = ( _velocity.y[idy * c_rowVelocity[1] + idx] + _velocity.y[(idy + 1) * c_rowVelocity[1] + idx] ) / 2.0f;

    real oldX = (real)idx + 0.5f - cvx * _timestep;
    real oldY = (real)idy + 0.5f - cvy * _timestep;

    if(oldX < 1.0f) oldX = 1.0f;
    if(oldX > c_gridSize[0]-1.0f) oldX = c_gridSize[0]-1.0f;
    if(oldY < 1.0f) oldY = 1.0f;
    if(oldY > c_gridSize[1]-1.0f) oldY = c_gridSize[1]-1.0f;

    int i0 = (int)(oldX-0.5f);
    int j0 = (int)(oldY-0.5f);
    int i1 = i0+1;
    int j1 = j0+1;

    real wL = (real)i1+0.5f-oldX;
    real wR = 1.0f-wL;
    real wB = (real)j1+0.5f-oldY;
    real wT = 1.0f-wB;

    _value[currentIdx] = wB*(wL*_value0[j0 * c_gridSize[0] + i0]
        +wR*_value0[j0 * c_gridSize[0] + i1])+
        wT*(wL*_value0[j1 * c_gridSize[0] + i0]
        +wR*_value0[j1 * c_gridSize[0] + i1]);
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_diffuseVelocity( tuple<real *> _previousVelocity, tuple<real *> _velocity, real _timestep, real _diffusion )
{
  // to be continued... still need to check boundary conditions
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  extern __shared__ real local_velocity[];

  real a = _diffusion * _timestep;

  int sIdx = threadIdx.y * blockDim.x + threadIdx.x;

  for(int k = 0; k < 20; k++)
  {
    if ( idx > 0 && idx < c_rowVelocity[0] - 1 &&
         idy > 0 && idy < c_columnVelocity[0] - 1 )
    {
      local_velocity[sIdx] = (_previousVelocity.x[idy * c_rowVelocity[0] + idx]+
          a*(_velocity.x[idy * c_rowVelocity[0] + (idx + 1)]+
          _velocity.x[idy * c_rowVelocity[0] + (idx - 1)]+
          _velocity.x[(idy + 1) * c_rowVelocity[0] + idx]+
          _velocity.x[(idy - 1) * c_rowVelocity[0] + idx])) / (4.0*a+1.0);
      __syncthreads();
      _velocity.x[idy * c_rowVelocity[0] + idx] = local_velocity[sIdx];
    }

    if ( idx > 0 && idx < c_rowVelocity[1] - 1 &&
         idy > 0 && idy < c_columnVelocity[1] - 1 )
    {
      local_velocity[sIdx] =
          (_previousVelocity.y[idy * c_rowVelocity[1] + idx]+
          a*(_velocity.y[idy * c_rowVelocity[1] + (idx + 1)]+
          _velocity.y[idy * c_rowVelocity[1] + (idx - 1)]+
          _velocity.y[(idy + 1) * c_rowVelocity[1] + idx]+
          _velocity.y[(idy - 1) * c_rowVelocity[1] + idx])) / (4.0*a+1.0);

      __syncthreads();
      _velocity.y[idy * c_rowVelocity[1] + idx] = local_velocity[sIdx];
    }

    __syncthreads();

    // boundary cases
    if ( idx > 0 && idx < c_rowVelocity[0] - 1 ) // rowsize
    {
      _velocity.x[idx] =  _velocity.x[idx + c_rowVelocity[0]];
      _velocity.x[idx + c_rowVelocity[0] * (c_columnVelocity[0]-1)] = _velocity.x[idx + c_rowVelocity[0] * (c_columnVelocity[0] - 2)];
    }

    if ( idx > 0 && idx < c_rowVelocity[1] - 1 )
    {
      _velocity.y[idx] = - _velocity.y[idx + c_rowVelocity[1]];
      _velocity.y[idx + c_rowVelocity[1] * (c_columnVelocity[1]-1)] = -_velocity.y[idx + c_rowVelocity[1] * (c_columnVelocity[1] - 2)];
    }


    if ( idx > 0 && idx < c_columnVelocity[0] - 1 ) // colsize
    {
      _velocity.x[idx * c_rowVelocity[0]] = -_velocity.x[idx * c_rowVelocity[0] + 1];
      _velocity.x[idx * c_rowVelocity[0] + ( c_rowVelocity[0] - 1)] = -_velocity.x[idx * c_rowVelocity[0] + (c_rowVelocity[0] - 2)];
    }

    if ( idx > 0 && idx < c_columnVelocity[1] - 1 ) // colsize
    {
      _velocity.y[idx * c_rowVelocity[1]] = _velocity.y[idx * c_rowVelocity[1] + 1];
      _velocity.y[idx * c_rowVelocity[1] + ( c_rowVelocity[1] - 1)] = _velocity.y[idx * c_rowVelocity[1] + (c_rowVelocity[1] - 2)];
    }
    __syncthreads();

    // corners
    if ( idx == 0 && idy == 0 )
    {
      _velocity.x[0] = ( _velocity.x[1] + _velocity.x[c_rowVelocity[0]] ) / 2;

      int dst = c_rowVelocity[0] - 1;
      int left = c_rowVelocity[0] - 2;
      int down = c_rowVelocity[0] + c_rowVelocity[0] - 1;
      _velocity.x[dst] = (_velocity.x[left] + _velocity.x[down])/2;

      int up = (c_columnVelocity[0] - 1) * c_rowVelocity[0] + 1;
      left = (c_columnVelocity[0] - 2) * c_rowVelocity[0];
      dst = (c_columnVelocity[0] - 1) * c_rowVelocity[0];
      _velocity.x[dst] = (_velocity.x[up] + _velocity.x[left])/2;

      dst = (c_columnVelocity[0] - 1) * c_rowVelocity[0] + (c_rowVelocity[0] -1);
      left = (c_columnVelocity[0] - 1) * c_rowVelocity[0] + (c_rowVelocity[0] - 2);
      up = (c_columnVelocity[0] - 2) * c_rowVelocity[0] + (c_rowVelocity[0] - 1);

      _velocity.x[dst] = ( _velocity.x[left] + _velocity.x[up] ) / 2;
    }

    if ( idx == 1 && idy == 1 )
    {
      _velocity.y[0] = ( _velocity.y[1] + _velocity.y[c_rowVelocity[1]] ) / 2;

      int dst = c_rowVelocity[1] - 1;
      int left = c_rowVelocity[1] - 2;
      int down = c_rowVelocity[1] + c_rowVelocity[1] - 1;
      _velocity.y[dst] = ( _velocity.y[left] + _velocity.y[down] )/2;

      int up = (c_columnVelocity[1] - 1) * c_rowVelocity[1] + 1;
      left = (c_columnVelocity[1] - 2) * c_rowVelocity[1];
      dst = (c_columnVelocity[1] - 1) * c_rowVelocity[1];
      _velocity.y[dst] = (_velocity.y[up] + _velocity.y[left])/2;

      dst = (c_columnVelocity[1] - 1) * c_rowVelocity[1] + (c_rowVelocity[1] -1);
      left = (c_columnVelocity[1] - 1) * c_rowVelocity[1] + (c_rowVelocity[1] - 2);
      up = (c_columnVelocity[1] - 2) * c_rowVelocity[1] + (c_rowVelocity[1] - 1);

      _velocity.y[dst] = ( _velocity.y[left] + _velocity.y[up] ) / 2;
    }
    __syncthreads();
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_diffuseCell( real * _previousDensity, real * _density, real _timestep, real _viscosity )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  extern __shared__ real local_density[];

  real a = _viscosity * _timestep;

  int sIdx = threadIdx.y * blockDim.x + threadIdx.x;
  for(int k = 0; k < 20; k++)
  {
    if ( idx > 0 && idx < c_gridSize[0] - 1 &&
         idy > 0 && idy < c_gridSize[1] - 1 )
    {
      local_density[sIdx] = (_previousDensity[idy * c_gridSize[0] + idx]+
          a*(_density[idy * c_gridSize[0] + idx+1]+_density[idy * c_gridSize[0] + idx-1]+
          _density[(idy + 1) * c_gridSize[0] + idx]+
          _density[(idy - 1) * c_gridSize[0] + idx])) / (4.0f*a+1.0f);
      __syncthreads();

      _density[idy * c_gridSize[0] + idx] = local_density[sIdx];
    }

    // TODO: boundary
    if ( idx > 0 && idx < c_gridSize[0] - 1 )
    {
      _density[idx] = _density[idx + c_gridSize[0]];
      _density[idx + c_gridSize[0] * (c_gridSize[1] - 1)] = _density[idx + c_gridSize[0] * (c_gridSize[1] - 2)];
    }

    if ( idx > 0 && idx < c_gridSize[1] - 1 )
    {
      _density[idx * c_gridSize[0]] = _density[idx * c_gridSize[0] + 1]; // set the first column on the left to be the same as the next
      _density[idx * c_gridSize[0] + ( c_gridSize[0] - 1)] = _density[idx * c_gridSize[0] + (c_gridSize[0] - 2)];
    }
    if ( idx == 0 )
    {
      _density[0] = ( _density[1] + _density[c_gridSize[0]] ) / 2;

      int dst = c_gridSize[0] - 1;
      int left = c_gridSize[0] - 2;
      int down = c_gridSize[0] + c_gridSize[0] - 1;
      _density[dst] = (_density[left] + _density[down]) / 2;

      int up = (c_gridSize[1] - 1) * c_gridSize[0] + 1;
      left = (c_gridSize[1] - 2) * c_gridSize[0];
      dst = (c_gridSize[1] - 1) * c_gridSize[0];
      _density[dst] = (_density[up] + _density[left])/2;

      dst = (c_gridSize[1] - 1) * c_gridSize[0] + (c_gridSize[0] -1);
      left = (c_gridSize[1] - 1) * c_gridSize[0] + (c_gridSize[0] - 2);
      up = (c_gridSize[1] - 2) * c_gridSize[0] + (c_gridSize[0] - 1);

      _density[dst] = ( _density[left] + _density[up] ) / 2;
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_addVelocity_x( real * _previousVelocity, real * _velocity )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ( idx < c_totVelocity[0] )
  {
    _velocity[idx] += _previousVelocity[idx];
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_addVelocity_y( real * _previousVelocity, real * _velocity )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if ( idx < c_totVelocity[1] )
  {
    _velocity[idx] += _previousVelocity[idx];
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void d_addDensity( real * _previousDensity, real * _density )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int gridSize = c_gridSize[0] * c_gridSize[1];

  if ( idx < gridSize )
  {
    _density[idx] += _previousDensity[idx];
  }
}

//----------------------------------------------------------------------------------------------------------------------
