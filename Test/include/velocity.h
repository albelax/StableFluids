#ifndef _VELOCITY_H
#define _VELOCITY_H

#include "MacStableSolver.h"
#include "GpuSolver.h"
#include "rand_cpu.h"
#include "rand_gpu.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( boundary, velocity_x )
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

  gpuSolver.setVelBoundary( 1 );
  cpuSolver.setVelBoundary( 1 );

  real * h_inputX = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.x, h_inputX, gpuSolver.m_totVelX );

  for ( int i = 0; i < cpuSolver.m_totVelX; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_velocity.x[i], h_inputX[i] );
  }

  free( h_inputX );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( boundary, velocity_y )
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

  gpuSolver.setVelBoundary( 2 );
  cpuSolver.setVelBoundary( 2 );

  real * h_inputY = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.y, h_inputY, gpuSolver.m_totVelY );

  for ( int i = 0; i < gpuSolver.m_totVelY; ++i )
  {
    EXPECT_FLOAT_EQ( cpuSolver.m_velocity.y[i], h_inputY[i] );
  }

  free( h_inputY );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( projection, velocity_x )
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

  real * h_velocity_x = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.x, h_velocity_x, gpuSolver.m_totVelX );

  int x = cpuSolver.m_totVelX;
  for ( int i = 0; i < x; ++i )
  {
    EXPECT_NEAR( cpuSolver.m_velocity.x[i], h_velocity_x[i], 0.5f); // should probably try to bring it down a bit...
  }
  free( h_velocity_x );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( projection, velocity_y )
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

  real * h_velocity_y = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.y, h_velocity_y, gpuSolver.m_totVelY );

  int x = cpuSolver.m_totVelY;
  for ( int i = 0; i < x; ++i )
  {
    EXPECT_NEAR( cpuSolver.m_velocity.y[i], h_velocity_y[i], 0.5f); // should probably try to bring it down a bit...
  }
  free( h_velocity_y );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( advect, velocity_x )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_velocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.y,  gpuSolver.m_totVelY );



  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  gpuSolver.copy( gpuSolver.m_previousVelocity.x, cpuSolver.m_previousVelocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_previousVelocity.y, cpuSolver.m_previousVelocity.y, gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.x, cpuSolver.m_velocity.x, gpuSolver.m_totVelX );

  gpuSolver.advectVelocity();
  cpuSolver.advectVel();

  real * h_velocity_x = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.x, h_velocity_x, gpuSolver.m_totVelX );

  int x = cpuSolver.m_totVelX;
  for ( int i = 0; i < x; ++i )
  {
    ASSERT_EQ( cpuSolver.m_velocity.x[i], h_velocity_x[i]);
  }
  free( h_velocity_x );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( advect, velocity_y )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.y,  gpuSolver.m_totVelY );
  Rand_GPU::randFloats( gpuSolver.m_velocity.y,  gpuSolver.m_totVelY );


  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  gpuSolver.copy( gpuSolver.m_previousVelocity.y, cpuSolver.m_previousVelocity.y, gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.y, cpuSolver.m_velocity.y, gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.x, cpuSolver.m_velocity.x, gpuSolver.m_totVelX );


  gpuSolver.advectVelocity();
  cpuSolver.advectVel();

  real * h_velocity_y = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.y, h_velocity_y, gpuSolver.m_totVelY );

  int x = cpuSolver.m_totVelY;
  for ( int i = 0; i < x; ++i )
  {
    EXPECT_NEAR( cpuSolver.m_velocity.y[i], h_velocity_y[i], 0.5);
  }
  free( h_velocity_y );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( velocity_x, diffuse ) // to be continued...
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.y,  gpuSolver.m_totVelY );
  Rand_GPU::randFloats( gpuSolver.m_velocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_velocity.y,  gpuSolver.m_totVelY );

  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  gpuSolver.copy( gpuSolver.m_previousVelocity.x, cpuSolver.m_previousVelocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_previousVelocity.y, cpuSolver.m_previousVelocity.y, gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.x, cpuSolver.m_velocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.y, cpuSolver.m_velocity.y, gpuSolver.m_totVelY );

  gpuSolver.diffuseVelocity();
  cpuSolver.diffuseVel();

  real * h_velocity_x = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.x, h_velocity_x, gpuSolver.m_totVelX );

  int x = cpuSolver.m_totVelX;
  for ( int i = 0; i < x; ++i )
  {
    EXPECT_NEAR( cpuSolver.m_velocity.x[i], h_velocity_x[i], 0.5f);
  }
  free( h_velocity_x );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( velocity_y, diffuse )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.y,  gpuSolver.m_totVelY );
  Rand_GPU::randFloats( gpuSolver.m_velocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_velocity.y,  gpuSolver.m_totVelY );


  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  gpuSolver.copy( gpuSolver.m_previousVelocity.x, cpuSolver.m_previousVelocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_previousVelocity.y, cpuSolver.m_previousVelocity.y, gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.x, cpuSolver.m_velocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.y, cpuSolver.m_velocity.y, gpuSolver.m_totVelY );


  gpuSolver.diffuseVelocity();
  cpuSolver.diffuseVel();

  real * h_velocity_y = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.y, h_velocity_y, gpuSolver.m_totVelY );

  int x = cpuSolver.m_totVelY;
  for ( int i = 0; i < x; ++i )
  {
    EXPECT_NEAR( cpuSolver.m_velocity.y[i], h_velocity_y[i], 0.5f);
  }
  free( h_velocity_y );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( inputVelocity, setVel0 )
{
  GpuSolver gpuSolver;
  gpuSolver.activate();
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_previousVelocity.y,  gpuSolver.m_totVelY );
  Rand_GPU::randFloats( gpuSolver.m_velocity.x,  gpuSolver.m_totVelX );
  Rand_GPU::randFloats( gpuSolver.m_velocity.y,  gpuSolver.m_totVelY );


  StableSolverCpu cpuSolver;
  cpuSolver.activate();

  gpuSolver.copy( gpuSolver.m_previousVelocity.x, cpuSolver.m_previousVelocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_previousVelocity.y, cpuSolver.m_previousVelocity.y, gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.x, cpuSolver.m_velocity.x, gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.y, cpuSolver.m_velocity.y, gpuSolver.m_totVelY );

  int r_x = random()%128;
  int r_y = random()%128;
  int r_x0 = random()%128;
  int r_y0 = random()%128;

  gpuSolver.setVel0(r_x, r_y, r_x0, r_y0);
  cpuSolver.setVel0(r_x, r_y, r_x0, r_y0);

  real * h_velocity_y = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelY );
  gpuSolver.copy( gpuSolver.m_velocity.y, h_velocity_y, gpuSolver.m_totVelY );

  real * h_velocity_x = (real *) malloc( sizeof( real ) * gpuSolver.m_totVelX );
  gpuSolver.copy( gpuSolver.m_velocity.x, h_velocity_x, gpuSolver.m_totVelX );

  int x = cpuSolver.m_totVelY;
  for ( int i = 0; i < x; ++i )
    EXPECT_NEAR( cpuSolver.m_velocity.y[i], h_velocity_y[i], 0.5f);

  x = cpuSolver.m_totVelX;
  for ( int i = 0; i < x; ++i )
    EXPECT_NEAR( cpuSolver.m_velocity.x[i], h_velocity_x[i], 0.5f);
  free( h_velocity_x );
}

////----------------------------------------------------------------------------------------------------------------------

#endif // _VELOCITY_H
