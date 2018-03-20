#include "GpuSolver.cuh"
#include "GpuSolver.h"
#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <sys/time.h>
#include <time.h>

//----------------------------------------------------------------------------------------------------------------------

GpuSolver::~GpuSolver()
{
  cudaFree( m_pvx );
  cudaFree( m_pvy );
  cudaFree( m_density );
  cudaFree( m_pressure );
  cudaFree( m_divergence );
  cudaFree( m_velocity.x );
  cudaFree( m_velocity.y );
  cudaFree( m_previousVelocity.x );
  cudaFree( m_previousVelocity.y );
  cudaFree( m_previousDensity );
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::setParameters()
{
  m_gridSize.x = 128;
  m_gridSize.y = 128;

  m_totCell = m_gridSize.x * m_gridSize.y;
  m_rowVelocity.x = m_gridSize.x + 1;
  m_rowVelocity.y = m_gridSize.x;

  m_columnVelocity.x = m_gridSize.y;
  m_columnVelocity.y = m_gridSize.y + 1;

  m_totVelX = m_rowVelocity.x * m_columnVelocity.x;
  m_totVelY = m_rowVelocity.y * m_columnVelocity.y;

  m_min.x = 0.0f;
  m_max.x = (float)m_gridSize.x;
  m_min.y = 0.0f;
  m_max.y = (float)m_gridSize.y;

  m_timeStep = 1.0f;
  m_diffusion = 0.0f;
  m_viscosity = 0.0f;
}

void GpuSolver::allocateArrays()
{
  cudaMalloc( &m_pvx, sizeof(tuple<float>) * m_totVelX );
  cudaMalloc( &m_pvy, sizeof(tuple<float>) * m_totVelY );
  cudaMalloc( &m_density, sizeof(float)*m_totCell );
  cudaMalloc( &m_pressure, sizeof(float)*m_totCell );
  cudaMalloc( &m_divergence, sizeof(float)*m_totCell );
  cudaMalloc( &m_velocity.x, sizeof(float)*m_totVelX );
  cudaMalloc( &m_velocity.y, sizeof(float)*m_totVelY );
  cudaMalloc( &m_previousVelocity.x, sizeof(float)*m_totVelX );
  cudaMalloc( &m_previousVelocity.y, sizeof(float)*m_totVelY );
  cudaMalloc( &m_previousDensity, sizeof(float)*m_totCell );
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::activate()
{
  setParameters();
  allocateArrays();
  // 1024 -> max threads per block, in this case it will fire 16 blocks
  int nBlocks = m_totVelX / 1024;
  int blockDim = 1024 / m_gridSize.x + 1; // 9 threads per block

  dim3 block(blockDim, blockDim); // block of (X,Y) threads
  dim3 grid(nBlocks, nBlocks); // grid 2x2 blocks

  setPvx<<<grid, block>>>( m_pvx, m_rowVelocity.x );
  setPvy<<<grid, block>>>( m_pvy, m_rowVelocity.y );

  cudaThreadSynchronize();

  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) printf("Error: %s\n", cudaGetErrorString(err));
  //  exportCSV( "gpu_pvx.csv", m_pvx, m_rowVelocity.x, m_columnVelocity.x );
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::reset()
{
  int threads = 1024;
  unsigned int densityBlocks = m_totCell / threads + 1;
  unsigned int xVelocityBlocks = m_totVelX / threads + 1;
  unsigned int yVelocityBlocks = m_totVelY / threads + 1;

  d_reset<<<densityBlocks, threads>>>(m_density, m_totCell);
  d_reset<<<xVelocityBlocks, threads>>>(m_velocity.x, m_totVelX);
  d_reset<<<yVelocityBlocks, threads>>>(m_velocity.y, m_totVelY);
  cudaThreadSynchronize();
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::cleanBuffer()
{
  int threads = 1024;
  unsigned int densityBlocks = m_totCell / threads + 1;
  unsigned int xVelocityBlocks = m_totVelX / threads + 1;
  unsigned int yVelocityBlocks = m_totVelY / threads + 1;

  d_reset<<<densityBlocks, threads>>>(m_previousDensity, m_totCell);
  d_reset<<<xVelocityBlocks, threads>>>(m_previousVelocity.x, m_totVelX);
  d_reset<<<yVelocityBlocks, threads>>>(m_previousVelocity.y, m_totVelY);
  cudaThreadSynchronize();
}
//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::setVelBoundary( int flag )
{
  if(flag == 1)
  {
    for(int i=1; i<=m_rowVelocity.x-2; ++i)
    {
      m_velocity.x[vxIdx(i, 0)] = m_velocity.x[vxIdx(i, 1)];
      m_velocity.x[vxIdx(i, m_columnVelocity.x-1)] = m_velocity.x[vxIdx(i, m_columnVelocity.x-2)];
    }
    for(int j=1; j<=m_columnVelocity.x-2; ++j)
    {
      m_velocity.x[vxIdx(0, j)] = -m_velocity.x[vxIdx(1, j)];
      m_velocity.x[vxIdx(m_rowVelocity.x-1, j)] = -m_velocity.x[vxIdx(m_rowVelocity.x-2, j)];
    }
    m_velocity.x[vxIdx(0, 0)] = (m_velocity.x[vxIdx(1, 0)]+m_velocity.x[vxIdx(0, 1)])/2;
    m_velocity.x[vxIdx(m_rowVelocity.x-1, 0)] = (m_velocity.x[vxIdx(m_rowVelocity.x-2, 0)]+m_velocity.x[vxIdx(m_rowVelocity.x-1, 1)])/2;
    m_velocity.x[vxIdx(0, m_columnVelocity.x-1)] = (m_velocity.x[vxIdx(1, m_columnVelocity.x-1)]+m_velocity.x[vxIdx(0, m_columnVelocity.x-2)])/2;
    m_velocity.x[vxIdx(m_rowVelocity.x-1, m_columnVelocity.x-1)] = (m_velocity.x[vxIdx(m_rowVelocity.x-2, m_columnVelocity.x-1)]+m_velocity.x[vxIdx(m_rowVelocity.x-1, m_columnVelocity.x-2)])/2;
  }

  //y-axis
  if(flag == 2)
  {
    for(int i=1; i<=m_rowVelocity.y-2; ++i)
    {
      m_velocity.y[vyIdx(i, 0)] = -m_velocity.y[vyIdx(i, 1)];
      m_velocity.y[vyIdx(i, m_columnVelocity.y-1)] = -m_velocity.y[vyIdx(i, m_columnVelocity.y-2)];
    }
    for(int j=1; j<=m_columnVelocity.y-2; ++j)
    {
      m_velocity.y[vyIdx(0, j)] = m_velocity.y[vyIdx(1, j)];
      m_velocity.y[vyIdx(m_rowVelocity.y-1, j)] = m_velocity.y[vyIdx(m_rowVelocity.y-2, j)];
    }
    m_velocity.y[vyIdx(0, 0)] = (m_velocity.y[vyIdx(1, 0)]+m_velocity.y[vyIdx(0, 1)])/2;
    m_velocity.y[vyIdx(m_rowVelocity.y-1, 0)] = (m_velocity.y[vyIdx(m_rowVelocity.y-2, 0)]+m_velocity.y[vyIdx(m_rowVelocity.y-1, 1)])/2;
    m_velocity.y[vyIdx(0, m_columnVelocity.y-1)] = (m_velocity.y[vyIdx(1, m_columnVelocity.y-1)]+m_velocity.y[vyIdx(0, m_columnVelocity.y-2)])/2;
    m_velocity.y[vyIdx(m_rowVelocity.y-1, m_columnVelocity.y-1)] = (m_velocity.y[vyIdx(m_rowVelocity.y-2, m_columnVelocity.y-1)]+m_velocity.y[vyIdx(m_rowVelocity.y-1, m_columnVelocity.y-2)])/2;
  }
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::exportCSV( std::string _file, tuple<float> * _t, int _sizeX, int _sizeY )
{
  std::ofstream out;
  out.open( _file );
  out.clear();
  int totSize = _sizeX * _sizeY;
  tuple<float> * result = (tuple<float> *) malloc( sizeof( tuple<float> ) * totSize );
  if( cudaMemcpy( result, _t, totSize * sizeof(tuple<float>), cudaMemcpyDeviceToHost) != cudaSuccess )
    exit(0);

  for(int i = 0; i < _sizeX; ++i)
  {
    for(int j = 0; j < _sizeY; ++j)
    {
      int idx = j * _sizeX + i;
      out << "( " << result[idx].x << ", " << result[idx].y << " )" << "; ";
    }
    out << "\n";
  }
  free( result );
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::copy( tuple<float> * _src, tuple<float> * _dst, int _size )
{
  if( cudaMemcpy( _dst, _src, _size * sizeof(tuple<float>), cudaMemcpyDeviceToHost) != cudaSuccess )
    exit(0);
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::copy( float * _src, float * _dst, int _size )
{
  if( cudaMemcpy( _dst, _src, _size * sizeof( float ), cudaMemcpyDeviceToHost) != cudaSuccess )
    exit(0);
}

//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
// KERNELS -------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

__global__ void setPvx( tuple<float> * _pvx, unsigned int _size )
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

__global__ void setPvy( tuple<float> * _pvy, unsigned int _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if ( idx < _size )
  {
    int i = idy * _size + idx;
    _pvy[i].x = (float)idx + 0.5f;
    _pvy[i].y = idy;
  }
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void vectorAdd( float *sum, float *A, float *B, size_t arrayLength )
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
