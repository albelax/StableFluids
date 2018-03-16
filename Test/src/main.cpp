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
  tuple<float> * p = (tuple<float> *) malloc( sizeof( tuple<float> ) * gps.m_totVelX );
  gps.copy( gps.m_pvx, p, gps.m_totVelX );

  StableSolverCpu cps;
  bool same = true;
  for ( int i = 0; i < cps.m_totVelX -1 ; ++i )
  {

    if ( p[i].x != cps.m_pvx[i].x || p[i].y != cps.m_pvx->y )
      same = false;
  }
  free( p );
  EXPECT_TRUE( same );
}

////----------------------------------------------------------------------------------------------------------------------
