#include "GpuSolverKernels.cuh"
#include "GpuSolver.h"
#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <sys/time.h>
#include <time.h>
#include "rand_gpu.h"

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
  int mul = 1;
  m_gridSize.x = 128 * mul;
  m_gridSize.y = 128 * mul;

  m_totCell = m_gridSize.x * m_gridSize.y;
  m_rowVelocity.x = m_gridSize.x + 1;
  m_rowVelocity.y = m_gridSize.x;

  m_columnVelocity.x = m_gridSize.y;
  m_columnVelocity.y = m_gridSize.y + 1;

  m_totVelX = m_rowVelocity.x * m_columnVelocity.x;
  m_totVelY = m_rowVelocity.y * m_columnVelocity.y;

  m_min.x = 0.0f;
  m_max.x = (real)m_gridSize.x;
  m_min.y = 0.0f;
  m_max.y = (real)m_gridSize.y;

  m_timeStep = 1.0f;
  m_diffusion = 0.0f;
  m_viscosity = 0.0f;
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::allocateArrays()
{
  cudaMalloc( &m_pvx, sizeof(tuple<real>) * m_totVelX );
  cudaMalloc( &m_pvy, sizeof(tuple<real>) * m_totVelY );
  cudaMalloc( &m_density, sizeof(real)*m_totCell );
  cudaMalloc( &m_pressure, sizeof(real)*m_totCell );
  cudaMalloc( &m_divergence, sizeof(real)*m_totCell );
  cudaMalloc( &m_velocity.x, sizeof(real)*m_totVelX );
  cudaMalloc( &m_velocity.y, sizeof(real)*m_totVelY );
  cudaMalloc( &m_previousVelocity.x, sizeof(real)*m_totVelX );
  cudaMalloc( &m_previousVelocity.y, sizeof(real)*m_totVelY );
  cudaMalloc( &m_previousDensity, sizeof(real)*m_totCell );
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

  d_setPvx<<<grid, block>>>( m_pvx, m_rowVelocity.x );
  d_setPvy<<<grid, block>>>( m_pvy, m_rowVelocity.y );

  //  cudaThreadSynchronize();

  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) printf("Error: %s\n", cudaGetErrorString(err));
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
  //  cudaThreadSynchronize();
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
}
//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::setVelBoundary( int flag )
{
  if(flag == 1)
  {
    int threads = 1024;
    unsigned int blocks = std::max( m_columnVelocity.x, m_rowVelocity.x ) / threads + 1;
    tuple<unsigned int> size;
    size.x = m_rowVelocity.x;
    size.y = m_columnVelocity.x;
    d_setVelBoundaryX<<< blocks, threads>>>( m_velocity.x, size );
  }

  else if(flag == 2)
  {
    int threads = 1024;
    unsigned int blocks = std::max( m_columnVelocity.y, m_rowVelocity.y ) / threads + 1;
    tuple<unsigned int> size;
    size.x = m_rowVelocity.y;
    size.y = m_columnVelocity.y;
    d_setVelBoundaryY<<< blocks, threads>>>( m_velocity.y, size );
  }
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::setCellBoundary(real * _value, tuple<unsigned int> _size )
{
  int threads = 1024;
  unsigned int blocks = std::max( _size.x, _size.y ) / threads + 1;

  d_setCellBoundary<<< blocks, threads>>>( _value, _size );
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::projection()
{
  unsigned int bins = 81 * sizeof(real);
  int nBlocks = m_totCell / 1024;
  int blockDim = 1024 / m_gridSize.x + 1; // 9 threads per block

  dim3 block(blockDim, blockDim); // block of (X,Y) threads
  dim3 grid(nBlocks, nBlocks); // grid 2x2 blocks

  d_divergenceStep<<<grid, block, bins>>>( m_pressure, m_divergence, m_velocity, m_rowVelocity,  m_gridSize );
  setCellBoundary( m_pressure, m_gridSize );
  setCellBoundary( m_divergence, m_gridSize );
  for(unsigned int k = 0; k < 20; k++)
  {
    d_projection<<<grid, block, bins>>>( m_pressure, m_divergence, m_gridSize );
    cudaThreadSynchronize();
  }
  setVelBoundary(1);
  setVelBoundary(2);
  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) printf("Projection Error: %s\n", cudaGetErrorString(err));
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::exportCSV( std::string _file, tuple<real> * _t, int _sizeX, int _sizeY )
{
  std::ofstream out;
  out.open( _file );
  out.clear();
  int totSize = _sizeX * _sizeY;
  tuple<real> * result = (tuple<real> *) malloc( sizeof( tuple<real> ) * totSize );
  if( cudaMemcpy( result, _t, totSize * sizeof(tuple<real>), cudaMemcpyDeviceToHost) != cudaSuccess )
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

void GpuSolver::copy( tuple<real> * _src, tuple<real> * _dst, int _size )
{
  if( cudaMemcpy( _dst, _src, _size * sizeof(tuple<real>), cudaMemcpyDeviceToHost) != cudaSuccess )
    exit(0);
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::copy( real * _src, real * _dst, int _size )
{
  if( cudaMemcpy( _dst, _src, _size * sizeof( real ), cudaMemcpyDeviceToHost) != cudaSuccess )
    exit(0);
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::gather( real * _value, unsigned int _size )
{
  int threads = 1024;
  unsigned int blocks = _size / threads + 1;

  real * d_values;
  cudaMalloc( &d_values, sizeof(real) * _size );
  if( cudaMemcpy( d_values, _value, _size * sizeof( real ), cudaMemcpyHostToDevice) != cudaSuccess )
    exit(0);

  unsigned int bins = 10;
  d_gather<<< blocks, threads, bins * sizeof(real)>>>( d_values, _size );
  //  cudaThreadSynchronize();

  copy( d_values, _value, _size );

}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::randomizeArrays()
{
  Rand_GPU::randFloats( m_pressure, m_totCell );
  Rand_GPU::randFloats( m_divergence, m_totCell );
  Rand_GPU::randFloats( m_velocity.x, m_totVelX );
  Rand_GPU::randFloats( m_velocity.y, m_totVelY );
}
