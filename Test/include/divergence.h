#ifndef DIVERGENCE_H
#define DIVERGENCE_H

#include "MacStableSolver.h"
#include "GpuSolver.h"
#include "rand_cpu.h"
#include "rand_gpu.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( divergenceBoundary, isEqual )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_divergence,  gpuSolver.m_totCell );

  StableSolverCpu cpuSolver;
  cpuSolver.activate();
  gpuSolver.copy( gpuSolver.m_divergence, cpuSolver.m_divergence, gpuSolver.m_totCell );

  tuple<unsigned int> t;
  t.x = gpuSolver.m_gridSize.x;
  t.y = gpuSolver.m_gridSize.y;

  gpuSolver.setCellBoundary( gpuSolver.m_divergence, t );
  cpuSolver.setCellBoundary( cpuSolver.m_divergence );


  real * h_inputD = (real *) malloc( sizeof( real ) * gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_divergence, h_inputD, gpuSolver.m_totCell );

  for ( int i = 0; i < cpuSolver.m_totCell; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_divergence[i], h_inputD[i] );
  }

  free( h_inputD );
}

////----------------------------------------------------------------------------------------------------------------------

#endif // DIVERGENCE_H
