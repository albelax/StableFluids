#include "GpuSolver.cuh"
#include "GpuSolver.h"
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>

//----------------------------------------------------------------------------------------------------------------------

GpuSolver::GpuSolver()
{
  init();
}


//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::init()
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

  //params
  m_timeStep = 1.0f;
  m_diffusion = 0.0f;
  m_viscosity = 0.0f;

  //  cudaMalloc( &m_density, sizeof(float)*m_totCell );
  //  cudaMalloc( &m_pressure, sizeof(float)*m_totCell );
  //  cudaMalloc( &m_divergence, sizeof(float)*m_totCell );
  //  cudaMalloc( &m_velocity.x, sizeof(float)*m_totVelX );
  //  cudaMalloc( &m_velocity.y, sizeof(float)*m_totVelY );
  //  cudaMalloc( &m_previousVelocity.x, sizeof(float)*m_totVelX );
  //  cudaMalloc( &m_previousVelocity.y, sizeof(float)*m_totVelY );
  //  cudaMalloc( &m_previousDensity, sizeof(float)*m_totCell );
  //  cudaMalloc( &m_pvy, sizeof(vec2<float>)*m_totVelY );

  int threads = 32;
  int threadsPerAxis = 16; // half of total threads if we fire a 2x2 block
  int blocks = m_totVelX / (threads * threads);
  dim3 block(threadsPerAxis, threadsPerAxis); // block of (X,Y) threads
  dim3 grid(blocks, blocks); // grid 2x2 blocks

  cudaMalloc( &m_pvx, sizeof(tuple<float>)*m_totVelX );

  setPvx<<<grid, block>>>( m_pvx, m_rowVelocity.x );

  cudaThreadSynchronize();
  m_result = (tuple<float> *)malloc(sizeof(tuple<float>)*m_totVelX);
  if( cudaMemcpy(m_result, m_pvx, m_totVelX * sizeof(tuple<float>), cudaMemcpyDeviceToHost) != cudaSuccess)
    exit(0);

  exportCSV("gpu_pvx.csv");
  free( m_result );
  cudaFree( m_pvx );
}

//----------------------------------------------------------------------------------------------------------------------

void GpuSolver::exportCSV( std::string _file )
{
  std::ofstream out;
  out.open( _file );
  out.clear();
  for(int i=0; i<m_rowVelocity.x; ++i)
  {
    for(int j=0; j<m_columnVelocity.x; ++j)
    {
      out << "( " << m_result[vxIdx(i, j)].x << ", " << m_result[vxIdx(i, j)].y << " )" << "; ";
    }
    out << "\n";
  }
}

//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
// KERNELS -------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

// pvx[i,j] = pvx[j * size +i]
// pvy[i,j] = pvy[j * size +i]
__global__ void setPvx( tuple<float> * _pvx, unsigned int _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  int i = idy * _size + idx;
  _pvx[i].x = idx;
  _pvx[i].y = idy + 0.5f;

}

//----------------------------------------------------------------------------------------------------------------------

__global__ void setPvy( tuple<float> * _pvy, unsigned int _size )
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  int i = idy * _size + idx;
  _pvy[i].x = (float) idx + 0.5f;
  _pvy[i].y = (float) idy;
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


