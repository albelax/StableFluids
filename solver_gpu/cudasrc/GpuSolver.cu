#include "GpuSolverKernels.cuh"
#include "GpuSolver.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>
#include "rand_gpu.h"
#include "parameters.h"

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
  m_gridSize.x = Common::gridWidth;
  m_gridSize.y = Common::gridHeight;
  m_totCell = Common::totCells;
  m_rowVelocity.x = Common::rowVelocityX;
  m_rowVelocity.y = Common::rowVelocityY;

  m_columnVelocity.x = Common::columnVelocityX;
  m_columnVelocity.y = Common::columnVelocityY;

  m_totVelX = Common::totHorizontalVelocity;
  m_totVelY = Common::totVerticalVelocity;

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

  // allocate constant memory
  unsigned int tmp_gridSize[] = { m_gridSize.x, m_gridSize.y };
  cudaMemcpyToSymbol(c_gridSize, tmp_gridSize, sizeof(unsigned int)*2 );

  unsigned int tmp_rowVelocity[] = { m_rowVelocity.x, m_rowVelocity.y };
  cudaMemcpyToSymbol(c_rowVelocity, tmp_rowVelocity, sizeof(int)*2,  0, cudaMemcpyHostToDevice );

  unsigned int tmp_columnVelocity[] = { m_columnVelocity.x, m_columnVelocity.y };
  cudaMemcpyToSymbol(c_columnVelocity, tmp_columnVelocity, sizeof(int)*2,  0, cudaMemcpyHostToDevice );

  unsigned int tmp_totVelocity[] = { m_totVelX, m_totVelY };
  cudaMemcpyToSymbol(c_totVelocity, tmp_totVelocity, sizeof(int)*2,  0, cudaMemcpyHostToDevice );
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

  d_setPvx<<<grid, block>>>( m_pvx );
  d_setPvy<<<grid, block>>>( m_pvy );

  //  cudaThreadSynchronize();

  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) printf("Error: %s\n", cudaGetErrorString(err));
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::reset()
{
  // will have to change this one,
  // the kenel launch overhead it too damn high
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
    d_setVelBoundaryX<<< blocks, threads>>>( m_velocity.x );
  }

  else if(flag == 2)
  {
    int threads = 1024;
    unsigned int blocks = std::max( m_columnVelocity.y, m_rowVelocity.y ) / threads + 1;
    d_setVelBoundaryY<<< blocks, threads>>>( m_velocity.y );
  }
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::setCellBoundary(real * _value, tuple<unsigned int> & _size )
{
  int threads = 1024;
  unsigned int blocks = std::max( _size.x, _size.y ) / threads + 1;

  d_setCellBoundary<<< blocks, threads>>>( _value );
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::projection()
{
  int nBlocks = m_totCell / 1024;
  int blockDim = 1024 / m_gridSize.x + 1; // 9 threads per block
  unsigned int bins = blockDim * blockDim * sizeof(real);

  dim3 block(blockDim, blockDim); // block of (X,Y) threads
  dim3 grid(nBlocks, nBlocks); // grid 2x2 blocks

  d_divergenceStep<<<grid, block, bins>>>( m_pressure, m_divergence, m_velocity );
  //  cudaThreadSynchronize();

  setCellBoundary( m_pressure, m_gridSize );
  setCellBoundary( m_divergence, m_gridSize );
  //  cudaThreadSynchronize();

  for(unsigned int k = 0; k < 20; k++)
  {
    d_projection<<<grid, block, bins>>>( m_pressure, m_divergence );
    //    cudaThreadSynchronize();

    setCellBoundary( m_pressure, m_gridSize );
    //    cudaThreadSynchronize();
  }

  d_velocityStep<<<grid, block, bins>>>( m_pressure, m_velocity );
  //  cudaThreadSynchronize();

  setVelBoundary(1);
  setVelBoundary(2);
  //  cudaThreadSynchronize();

  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) printf("Projection Error: %s\n", cudaGetErrorString(err));
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::advectVelocity()
{
  int nBlocks = m_totCell / 1024;
  int blockDim = 1024 / m_gridSize.x + 1;
  unsigned int bins = blockDim * blockDim * sizeof(real);

  dim3 block(blockDim, blockDim);
  dim3 grid(nBlocks, nBlocks);
  d_advectVelocity<<<grid, block, bins>>>( m_previousVelocity, m_velocity, m_pvx, m_pvy, m_timeStep );
  setVelBoundary(1);
  setVelBoundary(2);
  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) printf("Advection Error: %s\n", cudaGetErrorString(err));
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::advectCell()
{
  int nBlocks = m_totCell / 1024;
  int blockDim = 1024 / m_gridSize.x + 1;

  dim3 block(blockDim, blockDim);
  dim3 grid(nBlocks, nBlocks);
  d_advectCell<<<grid, block>>>( m_density, m_previousDensity, m_velocity, m_timeStep );
  setCellBoundary( m_density, m_gridSize );

  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) printf("Advection Error: %s\n", cudaGetErrorString(err));
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::diffuseVelocity()
{
  int nBlocks = m_totCell / 1024;
  int blockDim = 1024 / m_gridSize.x + 1;
  unsigned int bins = blockDim * blockDim * sizeof(real);

  dim3 block(blockDim, blockDim);
  dim3 grid(nBlocks, nBlocks);

//  cudaMemset( (void **)&m_velocity.x, 0, sizeof(real)*m_totVelX );
//  cudaMemset( (void **)&m_velocity.y, 0, sizeof(real)*m_totVelY );

  d_diffuseVelocity<<<grid, block, bins>>>( m_previousVelocity, m_velocity, m_timeStep, m_diffusion );

  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) printf("Diffusion Error: %s\n", cudaGetErrorString(err));
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::exportCSV( std::string _file, tuple<real> * _t, int _sizeX, int _sizeY )
{
  std::ofstream out;
  out.open( _file );
  out.clear();
  int totSize = _sizeX * _sizeY;
  tuple<real> * result = (tuple<real> *) malloc( sizeof( tuple<real> ) * totSize );
  if( cudaMemcpyAsync( result, _t, totSize * sizeof(tuple<real>), cudaMemcpyDeviceToHost) != cudaSuccess )
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

void GpuSolver::animVel()
{
  projection();

  if(m_diffusion > 0.0f)
  {
    SWAP(m_previousVelocity.x, m_velocity.x);
    SWAP(m_previousVelocity.y, m_velocity.y);
    diffuseVelocity();
  }

  SWAP(m_previousVelocity.x, m_velocity.x);
  SWAP(m_previousVelocity.y, m_velocity.y);
  advectVelocity();

  projection();
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::animDen()
{
  if(m_viscosity > 0.0f)
  {
    SWAP(m_previousDensity, m_density);
//    diffuseCell(m_density, m_previousDensity);
  }

  SWAP(m_previousDensity, m_density);
  advectCell();
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::copy( tuple<real> * _src, tuple<real> * _dst, int _size )
{
  if( cudaMemcpyAsync( _dst, _src, _size * sizeof(tuple<real>), cudaMemcpyDeviceToHost) != cudaSuccess )
    exit(0);
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::copy( real * _src, real * _dst, int _size )
{
  if( cudaMemcpyAsync( _dst, _src, _size * sizeof( real ), cudaMemcpyDeviceToHost) != cudaSuccess )
    exit(0);
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::copyToDevice( real * _src, real * _dst, int _size )
{
  if( cudaMemcpyAsync( _dst, _src, _size * sizeof( real ), cudaMemcpyHostToDevice ) != cudaSuccess )
    exit(0);
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::gather( real * _value, unsigned int _size )
{
  int threads = 1024;
  unsigned int blocks = _size / threads + 1;

  real * d_values;
  cudaMalloc( &d_values, sizeof(real) * _size );
  if( cudaMemcpyAsync( d_values, _value, _size * sizeof( real ), cudaMemcpyHostToDevice) != cudaSuccess )
    exit(0);

  unsigned int bins = 10;
  d_gather<<< blocks, threads, bins * sizeof(real)>>>( d_values, _size );

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

//----------------------------------------------------------------------------------------------------------------------
