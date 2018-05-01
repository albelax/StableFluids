#ifndef _GPUSOLVER_H
#define _GPUSOLVER_H

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include "tuple.h"
#include "Solver.h"

// friends the test functions with this class
#define TESTING 1

#if TESTING
#include <gtest/gtest.h>
#endif // TESTING

class GpuSolver : public Solver
{
public:
  GpuSolver() = default;
  ~GpuSolver();
  void activate() override;
  void reset() override;
  void cleanBuffer() override;

  void setVelBoundary( int flag );
  void setCellBoundary( real * _value, tuple<unsigned int> & _size );
  void setVel0(int i, int j, real _vx0, real _vy0) override;
  void setD0(int i, int j ) override;
  void addSource() override;
  void animVel() override;
  void animDen() override;
  void projection();
  void advectVelocity();
  void advectCell();
  void diffuseVelocity();
  void diffuseCell();
  void exportCSV( std::string _file, tuple<real> * _t, int _sizeX , int _sizeY );

  void gather( real * _value, unsigned int _size );
  void gather2D( real * _value, unsigned int _size );
  void randomizeArrays();
  const real * getDens() override;

private:
  void setParameters();
  void allocateArrays();
  // cpu
  real * m_cpuDensity;
  real * m_cpuPrevDensity;
  tuple<real *> m_cpuPreviousVelocity;
  // end cpu

  void copy( tuple<real> * _src, tuple<real> * _dst, int _size );
  void copy( real * _src, real * _dst, int _size );
  void copy( const real * _src, real * _dst, int _size );
  void copyToDevice( real * _src, real * _dst, int _size );

#if TESTING // I know friend is bad, but it is only to allow tesats to run :(
  FRIEND_TEST( pvx, isEqual );
  FRIEND_TEST( pvy, isEqual );
  FRIEND_TEST( resetDensity, isZero );
  FRIEND_TEST( resetVelocityX, isZero );
  FRIEND_TEST( resetVelocityY, isZero );
  FRIEND_TEST( boundary, velocity_x );
  FRIEND_TEST( boundary, velocity_y );
  FRIEND_TEST( densityBoundary, isEqual );
  FRIEND_TEST( boundary, pressure );
  FRIEND_TEST( divergenceBoundary, isEqual );
  FRIEND_TEST( gather, works );
  FRIEND_TEST( projection, pressure );
  FRIEND_TEST( projection, checkDivergence );
  FRIEND_TEST( projection, velocity_x );
  FRIEND_TEST( projection, velocity_y );
  FRIEND_TEST( advect, velocity_x );
  FRIEND_TEST( advect, velocity_y );
  FRIEND_TEST( velocity_x, diffuse );
  FRIEND_TEST( velocity_y, diffuse );
  FRIEND_TEST( inputVelocity, setVel0 );

#endif // TESTING
};

#endif // _GPUSOLVER_H
