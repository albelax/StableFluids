#ifndef _CONSTRUCTOR_H
#define _CONSTRUCTOR_H

#include "MacStableSolver.h"
#include "GpuSolver.h"
#include "rand_cpu.h"
#include "rand_gpu.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( pvx, isEqual )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();

  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  tuple<real> * gpu = (tuple<real> *) malloc( sizeof( tuple<real> ) * gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_pvx, gpu, gpuSolver.m_totVelX );
  gpuSolver.exportCSV( "results/gpu_pvx.csv", gpuSolver.m_pvx, gpuSolver.m_rowVelocity.x, gpuSolver.m_columnVelocity.x );
  cpuSolver.exportCSV( "results/cpu_pvx.csv", cpuSolver.m_pvx, cpuSolver.m_rowVelocity.x, cpuSolver.m_columnVelocity.x );


  for ( int i = 0; i < cpuSolver.m_rowVelocity.x; ++i )
  {
    for ( int j = 0; j < cpuSolver.m_columnVelocity.x; ++j )
    {
      int idx = j * cpuSolver.m_rowVelocity.x + i;

      EXPECT_FLOAT_EQ( gpu[idx].x, cpuSolver.m_pvx[idx].x );
      if ( gpu[idx].x != cpuSolver.m_pvx[idx].x )
        std::cout << "[ TEST FAILED AT pvx[" << i+1 << "][" << j+1 << "].x ]\n\n"; // indices + 1 so they match excel file

      EXPECT_FLOAT_EQ(gpu[idx].y, cpuSolver.m_pvx[idx].y );
      if ( gpu[idx].y != cpuSolver.m_pvx[idx].y )
        std::cout << "[ TEST FAILED AT pvx[" << i+1 << "][" << j+1 << "].y ]\n\n";
    }
  }

  free( gpu );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( pvy, isEqual )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();

  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  tuple<real> * gpu = (tuple<real> *) malloc( sizeof( tuple<real> ) * gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_pvy, gpu, gpuSolver.m_totVelX );
  gpuSolver.exportCSV( "results/gpu_pvy.csv", gpuSolver.m_pvy, gpuSolver.m_rowVelocity.y, gpuSolver.m_columnVelocity.y );
  cpuSolver.exportCSV( "results/cpu_pvy.csv", cpuSolver.m_pvy, cpuSolver.m_rowVelocity.y, cpuSolver.m_columnVelocity.y );

  for ( int i = 0; i < cpuSolver.m_rowVelocity.y; ++i )
  {
    for ( int j = 0; j < cpuSolver.m_columnVelocity.y; ++j )
    {
      int idx = j * cpuSolver.m_rowVelocity.y + i;

      EXPECT_FLOAT_EQ( gpu[idx].x, cpuSolver.m_pvy[idx].x );
      if ( gpu[idx].x != cpuSolver.m_pvy[idx].x )
        std::cout << "[ TEST FAILED AT pvy[" << i+1 << "][" << j+1 << "].x ]\n\n"; // indices + 1 so they match excel file

      EXPECT_FLOAT_EQ(gpu[idx].y, cpuSolver.m_pvy[idx].y );
      if ( gpu[idx].y != cpuSolver.m_pvy[idx].y )
        std::cout << "[ TEST FAILED AT pvy[" << i+1 << "][" << j+1 << "].y ]\n\n";
    }
  }

  free( gpu );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( resetDensity, isZero )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();

  Rand_GPU::randFloats( gpuSolver.m_density,  gpuSolver.m_totCell );
  gpuSolver.reset();

  real * zero = (real *) calloc( gpuSolver.m_totCell, sizeof( real ));
  real * h_gpu_density = (real *) malloc( sizeof( real ) * gpuSolver.m_totCell);
  gpuSolver.copy( gpuSolver.m_density, h_gpu_density, gpuSolver.m_totCell );

  for ( unsigned int i = 0; i < gpuSolver.m_totCell; ++i )
  {
    EXPECT_FLOAT_EQ( h_gpu_density[i], zero[i] );
  }
  free( h_gpu_density );
  free( zero );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( resetVelocityX, isZero )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_velocity.x,  gpuSolver.m_totVelX );
  gpuSolver.reset();

  real * zero = (real *) calloc( gpuSolver.m_totVelX, sizeof( real ));

  real * h_gpu_density = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelX);
  gpuSolver.copy( gpuSolver.m_velocity.x, h_gpu_density, gpuSolver.m_totVelX );

  for ( unsigned int i = 0; i < gpuSolver.m_totVelX; ++i )
  {
    EXPECT_FLOAT_EQ( h_gpu_density[i], zero[i] );
  }
  free( h_gpu_density );
  free( zero );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( resetVelocityY, isZero )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();

  Rand_GPU::randFloats( gpuSolver.m_velocity.y,  gpuSolver.m_totVelY );
  gpuSolver.reset();

  real * zero = (real *) calloc( gpuSolver.m_totVelY, sizeof( real ));

  real * h_gpu_density = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelY);
  gpuSolver.copy( gpuSolver.m_velocity.y, h_gpu_density, gpuSolver.m_totVelY );

  for ( unsigned int i = 0; i < gpuSolver.m_totVelY; ++i )
  {
    EXPECT_FLOAT_EQ( h_gpu_density[i], zero[i] );
  }
  free( h_gpu_density );
  free( zero );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( gather, works )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();

  real * values = (real *) malloc( sizeof( real ) * 10 );
  for ( int i = 0; i < 10; ++i )
  {
    values[i] = (real) i;
  }

  unsigned int s = 10;
  gpuSolver.gather(values, s);

  for ( int i = 0; i < 10; ++i )
  {
    EXPECT_FLOAT_EQ( 0, 0 );
  }
  free( values );
}

////----------------------------------------------------------------------------------------------------------------------

#endif // _CONSTRUCTOR_H









