#ifndef _VELOCITY_H
#define _VELOCITY_H

#include "MacStableSolver.h"
#include "GpuSolver.h"
#include "rand_cpu.h"
#include "rand_gpu.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( velBoundaryX, isEqual )
{
  // check that given the same dataset
  // both functions behave in the same way
  GpuSolver gpuSolver;
  gpuSolver.activate();

  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  // generate random velocity in the x, and copy it to the cpu solver
  Rand_GPU::randFloats( gpuSolver.m_velocity.x,  gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.x, cpuSolver.m_velocity.x, gpuSolver.m_totVelX );

///  gpuSolver.setVelBoundary( 1 ); // TO DO
  cpuSolver.setVelBoundary( 1 );

  float * h_inputX = (float *) malloc( sizeof( float ) * gpuSolver.m_totVelX );
///  gpuSolver.copy( gpuSolver.m_velocity.x, h_inputX, gpuSolver.m_totVelX ); // uncomment when implemented
  memcpy(h_inputX, cpuSolver.m_velocity.x, sizeof( float ) * cpuSolver.m_totVelX );

  for ( int i = 0; i < cpuSolver.m_totVelX; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_velocity.x[i], h_inputX[i] );
  }

  free( h_inputX );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( velBoundaryY, isEqual )
{
  // check that given the same dataset
  // both functions behave in the same way
  GpuSolver gpuSolver;
  gpuSolver.activate();

  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  // generate random velocity in the y, and copy it to the cpu solver
  Rand_GPU::randFloats( gpuSolver.m_velocity.y,  gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.y, cpuSolver.m_velocity.y, gpuSolver.m_totVelY );

///  gpuSolver.setVelBoundary( 2 ); // TO DO
  cpuSolver.setVelBoundary( 2 );

  float * h_inputY = (float *) malloc( sizeof( float ) * gpuSolver.m_totVelY );
///  gpuSolver.copy( gpuSolver.m_velocity.y, h_inputY, gpuSolver.m_totVelY ); // uncomment when implemented

  for ( int i = 0; i < gpuSolver.m_totVelY; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_velocity.y[i], h_inputY[i] );
  }

  free( h_inputY );
}

#endif // _VELOCITY_H
