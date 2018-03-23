#ifndef _GPUSOLVER_H
#define _GPUSOLVER_H

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include "tuple.h"

#define SWAP(value0,value) { real *tmp = value0; value0 = value; value = tmp; }
#define TESTING 1

#if TESTING
#include <gtest/gtest.h>
#endif // TESTING

class GpuSolver
{
public:
  GpuSolver() = default;
  ~GpuSolver();
  void activate();
  void reset();
  void cleanBuffer();
  void setTimestep( real _timeStep ) { m_timeStep = _timeStep; }
  void setDiffusion( real _diffusion ) { m_diffusion = _diffusion; }
  void setViscosity( real _viscosity ) { m_viscosity = _viscosity; }
  void setDensity( real _density ) { m_inputDensity = _density; }

  void setVelBoundary( int flag );
  void setCellBoundary( real * _value, tuple<unsigned int> _size );
  void projection();
  int vxIdx(int i, int j){ return j*m_rowVelocity.x+i; }
  int vyIdx(int i, int j){ return j*m_rowVelocity.y+i; }
  int cIdx(int i, int j){ return j*m_gridSize.x+i; }
  void exportCSV( std::string _file, tuple<real> * _t, int _sizeX , int _sizeY );

  void gather( real * _value, unsigned int _size );
  void gather2D( real * _value, unsigned int _size );
  void randomizeArrays();
private:
  void setParameters();
  void allocateArrays();
  unsigned int m_totCell;
  unsigned int m_totVelX;
  unsigned int m_totVelY;
  real m_timeStep;
  real m_diffusion;
  real m_viscosity;
  real m_inputDensity;

  real * m_density;
  real * m_previousDensity;
  real * m_divergence;
  real * m_pressure;
  tuple<unsigned int> m_gridSize;
  tuple<unsigned int> m_rowVelocity;
  tuple<unsigned int> m_columnVelocity;
  tuple<real> m_min;
  tuple<real> m_max;
  /// \brief velocity, stores to pointers to chunks of memory storing the velocities in x and y
  tuple<real *> m_velocity;
  tuple<real *> m_previousVelocity;
  tuple<real> * m_pvx;
  tuple<real> * m_pvy;
  void copy( tuple<real> * _src, tuple<real> * _dst, int _size );
  void copy( real * _src, real * _dst, int _size );

#if TESTING // I know friend is bad, but it is only to allow tesats to run :(
  FRIEND_TEST( pvx, isEqual );
  FRIEND_TEST( pvy, isEqual );
  FRIEND_TEST( resetDensity, isZero );
  FRIEND_TEST( resetVelocityX, isZero );
  FRIEND_TEST( resetVelocityY, isZero );
  FRIEND_TEST( velBoundaryX, isEqual );
  FRIEND_TEST( velBoundaryY, isEqual );
  FRIEND_TEST( densityBoundary, isEqual );
  FRIEND_TEST( pressureBoundary, isEqual );
  FRIEND_TEST( divergenceBoundary, isEqual );
  FRIEND_TEST( gather, works );
  FRIEND_TEST( projection, checkPressure );
  FRIEND_TEST( projection, checkDivergence );
  FRIEND_TEST( projection, checkVelocity_x );
  FRIEND_TEST( projection, checkVelocity_y );
#endif // TESTING
};

#endif // _GPUSOLVER_H
