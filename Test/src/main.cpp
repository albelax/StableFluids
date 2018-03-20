#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include "GpuSolver.h"
#include "MacStableSolver.h"
#include "rand_cpu.h"
#include "rand_gpu.h"

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

////----------------------------------------------------------------------------------------------------------------------

TEST( pvx, isEqual )
{
  GpuSolver gps;
  gps.activate();

  StableSolverCpu cpu;
  cpu.activate();

  tuple<float> * gpu = (tuple<float> *) malloc( sizeof( tuple<float> ) * gps.m_totVelX );
  gps.copy( gps.m_pvx, gpu, gps.m_totVelX );
  gps.exportCSV( "results/gpu_pvx.csv", gps.m_pvx, gps.m_rowVelocity.x, gps.m_columnVelocity.x );
  cpu.exportCSV( "results/cpu_pvx.csv", cpu.m_pvx, cpu.m_rowVelocity.x, cpu.m_columnVelocity.x );


  for ( int i = 0; i < cpu.m_rowVelocity.x; ++i )
  {
    for ( int j = 0; j < cpu.m_columnVelocity.x; ++j )
    {
      int idx = j * cpu.m_rowVelocity.x + i;

      EXPECT_FLOAT_EQ( gpu[idx].x, cpu.m_pvx[idx].x );
      if ( gpu[idx].x != cpu.m_pvx[idx].x )
        std::cout << "[ TEST FAILED AT pvx[" << i+1 << "][" << j+1 << "].x ]\n\n"; // indices + 1 so they match excel file

      EXPECT_FLOAT_EQ(gpu[idx].y, cpu.m_pvx[idx].y );
      if ( gpu[idx].y != cpu.m_pvx[idx].y )
        std::cout << "[ TEST FAILED AT pvx[" << i+1 << "][" << j+1 << "].y ]\n\n";
    }
  }

  free( gpu );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( pvy, isEqual )
{
  GpuSolver gps;
  gps.activate();

  StableSolverCpu cpu;
  cpu.activate();

  tuple<float> * gpu = (tuple<float> *) malloc( sizeof( tuple<float> ) * gps.m_totVelX );
  gps.copy( gps.m_pvy, gpu, gps.m_totVelX );
  gps.exportCSV( "results/gpu_pvy.csv", gps.m_pvy, gps.m_rowVelocity.y, gps.m_columnVelocity.y );
  cpu.exportCSV( "results/cpu_pvy.csv", cpu.m_pvy, cpu.m_rowVelocity.y, cpu.m_columnVelocity.y );

  for ( int i = 0; i < cpu.m_rowVelocity.y; ++i )
  {
    for ( int j = 0; j < cpu.m_columnVelocity.y; ++j )
    {
      int idx = j * cpu.m_rowVelocity.y + i;

      EXPECT_FLOAT_EQ( gpu[idx].x, cpu.m_pvy[idx].x );
      if ( gpu[idx].x != cpu.m_pvy[idx].x )
        std::cout << "[ TEST FAILED AT pvy[" << i+1 << "][" << j+1 << "].x ]\n\n"; // indices + 1 so they match excel file

      EXPECT_FLOAT_EQ(gpu[idx].y, cpu.m_pvy[idx].y );
      if ( gpu[idx].y != cpu.m_pvy[idx].y )
        std::cout << "[ TEST FAILED AT pvy[" << i+1 << "][" << j+1 << "].y ]\n\n";
    }
  }

  free( gpu );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( resetDensity, isZero )
{
  GpuSolver gps;
  gps.activate();

  Rand_GPU::randFloats( gps.m_density,  gps.m_totCell );
  gps.reset();

  float * zero = (float *) calloc( gps.m_totCell, sizeof( float ));
  float * h_gpu_density = (float *) malloc( sizeof( float ) * gps.m_totCell);
  gps.copy( gps.m_density, h_gpu_density, gps.m_totCell );

  for ( int i = 0; i < gps.m_totCell; ++i )
  {
    EXPECT_FLOAT_EQ( h_gpu_density[i], zero[i] );
  }
  free( h_gpu_density );
  free( zero );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( resetVelocityX, isZero )
{
  GpuSolver gps;
  gps.activate();
  Rand_GPU::randFloats( gps.m_velocity.x,  gps.m_totVelX );
  gps.reset();

  float * zero = (float *) calloc( gps.m_totVelX, sizeof( float ));

  float * h_gpu_density = (float *) malloc( sizeof( float ) * gps.m_totVelX);
  gps.copy( gps.m_velocity.x, h_gpu_density, gps.m_totVelX );

  for ( int i = 0; i < gps.m_totVelX; ++i )
  {
    EXPECT_FLOAT_EQ( h_gpu_density[i], zero[i] );
  }
  free( h_gpu_density );
  free( zero );
}

////----------------------------------------------------------------------------------------------------------------------

TEST( resetVelocityY, isZero )
{
  GpuSolver gps;
  gps.activate();

  Rand_GPU::randFloats( gps.m_velocity.y,  gps.m_totVelY );
  gps.reset();

  float * zero = (float *) calloc( gps.m_totVelY, sizeof( float ));

  float * h_gpu_density = (float *) malloc( sizeof( float ) * gps.m_totVelY);
  gps.copy( gps.m_velocity.y, h_gpu_density, gps.m_totVelY );

  for ( int i = 0; i < gps.m_totVelY; ++i )
  {
    EXPECT_FLOAT_EQ( h_gpu_density[i], zero[i] );
  }
  free( h_gpu_density );
  free( zero );
}

////----------------------------------------------------------------------------------------------------------------------
