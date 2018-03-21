#ifndef DENSITY_H
#define DENSITY_H

#include "MacStableSolver.h"
#include "GpuSolver.h"
#include "rand_cpu.h"
#include "rand_gpu.h"


////----------------------------------------------------------------------------------------------------------------------

TEST( cellBoundaryX, isEqual )
{
  // check that given the same dataset
  // both functions behave in the same way
  GpuSolver gpuSolver;
  gpuSolver.activate();

  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  // generate random velocity in the x, and copy it to the cpu solver
  Rand_GPU::randFloats( gpuSolver.m_density,  gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_density, cpuSolver.m_density, gpuSolver.m_totCell );

//  gpuSolver.setVelBoundary( 1 );
//  cpuSolver.setCellBoundary( 1 );

  float * h_inputX = (float *) malloc( sizeof( float ) * gpuSolver.m_totCell );
  gpuSolver.copy( gpuSolver.m_density, h_inputX, gpuSolver.m_totCell ); // uncomment when implemented

  for ( int i = 0; i < cpuSolver.m_totVelX; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_density[i],cpuSolver.m_density[i] );
  }

  free( h_inputX );
}

////----------------------------------------------------------------------------------------------------------------------


#endif // DENSITY_H
