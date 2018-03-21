#ifndef DENSITY_H
#define DENSITY_H

#include "MacStableSolver.h"
#include "GpuSolver.h"
#include "rand_cpu.h"
#include "rand_gpu.h"


////----------------------------------------------------------------------------------------------------------------------

TEST( cellBoundaryX, isEqual )
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


  float * h_inputD = (float *) malloc( sizeof( float ) * gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_density, h_inputD, gpuSolver.m_totCell );

  for ( int i = 0; i < cpuSolver.m_totCell; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_density[i], h_inputD[i] );
  }

  free( h_inputD );
}

////----------------------------------------------------------------------------------------------------------------------


#endif // DENSITY_H
