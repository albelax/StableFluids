#ifndef _GPUSOLVER_H
#define _GPUSOLVER_H

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include "tuple.h"

#define SWAP(value0,value) { real *tmp = value0; value0 = value; value = tmp; }

// friends the test functions with this class
#define TESTING 1
// testing gpu functions with the gpu solver
#define CROSS_TESTING 1
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
  void setCellBoundary( real * _value, tuple<unsigned int> & _size );
  void setVel0(int i, int j, real _vx0, real _vy0);
  void setD0(int i, int j );
  void projection();
  void advectVelocity();
  void advectCell();
  void diffuseVelocity();
  void diffuseCell();
  void addSource();
  void animVel();
  void animDen();

  int vxIdx(int i, int j) const { return j*m_rowVelocity.x+i; }
  int vyIdx(int i, int j) const { return j*m_rowVelocity.y+i; }
  int cIdx(int i, int j) const { return j*m_gridSize.x+i; }
  int getRowCell() const { return m_gridSize.x; }
  int getColCell() const { return m_gridSize.y; }
  int getTotCell() const { return m_totCell; }
  int getRowVelX() const { return m_rowVelocity.x; }
  int getcolVelX() const { return m_columnVelocity.x; }
  int getTotVelX() const { return m_totVelX; }
  int getRowVelY() const { return m_rowVelocity.y; }
  int getColVelY() const { return m_columnVelocity.y; }
  int getTotVelY() const { return m_totVelY; }
  void exportCSV( std::string _file, tuple<real> * _t, int _sizeX , int _sizeY );

  void gather( real * _value, unsigned int _size );
  void gather2D( real * _value, unsigned int _size );
  void randomizeArrays();

#if !CROSS_TESTING
private:
#endif

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
  real * m_cpuDensity;
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
  real * getDensity();
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

#endif // TESTING
};

#endif // _GPUSOLVER_H
