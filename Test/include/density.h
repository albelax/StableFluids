#ifndef DENSITY_H
#define DENSITY_H

#include "MacStableSolver.h"
#include "GpuSolver.h"
#include "rand_cpu.h"
#include "rand_gpu.h"


////----------------------------------------------------------------------------------------------------------------------

TEST( densityBoundary, isEqual )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_density,  gpuSolver.m_totCell );

  StableSolverCpu cpuSolver;
  cpuSolver.activate();
  gpuSolver.copy( gpuSolver.m_density, cpuSolver.m_density, gpuSolver.m_totCell );

  tuple<unsigned int> t;
  t.x = gpuSolver.m_gridSize.x;
  t.y = gpuSolver.m_gridSize.y;

  gpuSolver.setCellBoundary( gpuSolver.m_density, t );
  cpuSolver.setCellBoundary( cpuSolver.m_density );


  real * h_inputD = (real *) malloc( sizeof( real ) * gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_density, h_inputD, gpuSolver.m_totCell );

  for ( int i = 0; i < cpuSolver.m_totCell; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_density[i], h_inputD[i] );
  }

  free( h_inputD );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( advectDensity, isEqual )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_previousDensity,  gpuSolver.m_totCell );
  Rand_GPU::randFloats( gpuSolver.m_density,  gpuSolver.m_totCell );
  Rand_GPU::randFloats( gpuSolver.m_velocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_velocity.y,  gpuSolver.m_totVelY );

  StableSolverCpu cpuSolver;
  cpuSolver.activate();
  gpuSolver.copy( gpuSolver.m_previousDensity, cpuSolver.m_previousDensity, gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_density, cpuSolver.m_density, gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_velocity.x, cpuSolver.m_velocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.y, cpuSolver.m_velocity.y, gpuSolver.m_totVelY );

  tuple<unsigned int> t;
  t.x = gpuSolver.m_gridSize.x;
  t.y = gpuSolver.m_gridSize.y;

  gpuSolver.advectCell();
  cpuSolver.advectCell( cpuSolver.m_density, cpuSolver.m_previousDensity );


  real * h_inputD = (real *) malloc( sizeof( real ) * gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_density, h_inputD, gpuSolver.m_totCell );

  for ( int i = 0; i < cpuSolver.m_totCell; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_density[i], h_inputD[i] );
  }

  free( h_inputD );
}

////----------------------------------------------------------------------------------------------------------------------

#endif // DENSITY_H
