#ifndef PRESSURE_H
#define PRESSURE_H

#include "MacStableSolver.h"
#include "GpuSolver.h"
#include "rand_cpu.h"
#include "rand_gpu.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( boundary, pressure )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_pressure,  gpuSolver.m_totCell );

  StableSolverCpu cpuSolver;
  cpuSolver.activate();
  gpuSolver.copy( gpuSolver.m_pressure, cpuSolver.m_pressure, gpuSolver.m_totCell );

  tuple<unsigned int> t;
  t.x = gpuSolver.m_gridSize.x;
  t.y = gpuSolver.m_gridSize.y;

  gpuSolver.setCellBoundary( gpuSolver.m_pressure, t );
  cpuSolver.setCellBoundary( cpuSolver.m_pressure );


  real * h_inputD = (real *) malloc( sizeof( real ) * gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_pressure, h_inputD, gpuSolver.m_totCell );

  for ( int i = 0; i < cpuSolver.m_totCell; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_pressure[i], h_inputD[i] );
  }

  free( h_inputD );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( projection, pressure )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();

  Rand_GPU::randFloats( gpuSolver.m_pressure,  gpuSolver.m_totCell );
  Rand_GPU::randFloats( gpuSolver.m_divergence,  gpuSolver.m_totCell );
  Rand_GPU::randFloats( gpuSolver.m_velocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_velocity.y,  gpuSolver.m_totVelY );

  StableSolverCpu cpuSolver;
  cpuSolver.activate();
  gpuSolver.copy( gpuSolver.m_pressure, cpuSolver.m_pressure, gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_divergence, cpuSolver.m_divergence, gpuSolver.m_totCell );

  gpuSolver.copy( gpuSolver.m_velocity.x, cpuSolver.m_velocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.y, cpuSolver.m_velocity.y, gpuSolver.m_totVelY );

  gpuSolver.projection();
  cpuSolver.projection();

  real * h_pressure = (real *) malloc( sizeof( real ) * gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_pressure, h_pressure, gpuSolver.m_totCell );

  int x = cpuSolver.m_totCell;
  for ( int i = 0; i < x; ++i )
  {
    ASSERT_NEAR( cpuSolver.m_pressure[i], h_pressure[i], 0.25f ); // should probably try to bring it down a bit...
  }
  free( h_pressure );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( projection, checkDivergence )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();

  Rand_GPU::randFloats( gpuSolver.m_pressure,  gpuSolver.m_totCell );
  Rand_GPU::randFloats( gpuSolver.m_divergence,  gpuSolver.m_totCell );
  Rand_GPU::randFloats( gpuSolver.m_velocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_velocity.y,  gpuSolver.m_totVelY );

  StableSolverCpu cpuSolver;
  cpuSolver.activate();
  gpuSolver.copy( gpuSolver.m_pressure, cpuSolver.m_pressure, gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_divergence, cpuSolver.m_divergence, gpuSolver.m_totCell );

  gpuSolver.copy( gpuSolver.m_velocity.x, cpuSolver.m_velocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.y, cpuSolver.m_velocity.y, gpuSolver.m_totVelY );

  gpuSolver.projection();
  cpuSolver.projection();


  real * h_divergence = (real *) malloc( sizeof( real ) * gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_divergence, h_divergence, gpuSolver.m_totCell );

  int x = cpuSolver.m_totCell;
  for ( int i = 0; i < x; ++i )
  {
    ASSERT_NEAR( cpuSolver.m_divergence[i], h_divergence[i], 0.25f ); // should probably try to bring it down a bit...
  }
  free( h_divergence );
}

////----------------------------------------------------------------------------------------------------------------------

#endif // PRESSURE_H
