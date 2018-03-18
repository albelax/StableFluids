#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <gtest/gtest.h>
#include <fstream>
#include <random>
#include <iostream>
#include <random>
#include "GpuSolver.h"
#include "MacStableSolver.h"

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

////----------------------------------------------------------------------------------------------------------------------

TEST( pvx, isEqual )
{
  GpuSolver gps;
  StableSolverCpu cpu;
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
  StableSolverCpu cpu;
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
