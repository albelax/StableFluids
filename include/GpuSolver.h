#ifndef _GPUSOLVER_H
#define _GPUSOLVER_H

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include "tuple.h"

#define SWAP(value0,value) { float *tmp = value0; value0 = value; value = tmp; }
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
  void setTimestep( float _timeStep ) { m_timeStep = _timeStep; }
  void setDiffusion( float _diffusion ) { m_diffusion = _diffusion; }
  void setViscosity( float _viscosity ) { m_viscosity = _viscosity; }
  void setDensity( float _density ) { m_inputDensity = _density; }

  void setVelBoundary( int flag );
  void setCellBoundary( float * _value, tuple<unsigned int> _size );
  void projection();
  int vxIdx(int i, int j){ return j*m_rowVelocity.x+i; }
  int vyIdx(int i, int j){ return j*m_rowVelocity.y+i; }
  int cIdx(int i, int j){ return j*m_gridSize.x+i; }
  void exportCSV( std::string _file, tuple<float> * _t, int _sizeX , int _sizeY );

  void gather( float * _value, unsigned int _size );
  void gather2D( float * _value, unsigned int _size );
private:
  void setParameters();
  void allocateArrays();
  unsigned int m_totCell;
  unsigned int m_totVelX;
  unsigned  int m_totVelY;
  float m_timeStep;
  float m_diffusion;
  float m_viscosity;
  float m_inputDensity;

  float * m_density;
  float * m_previousDensity;
  float * m_divergence;
  float * m_pressure;
  tuple<unsigned int> m_gridSize;
  tuple<unsigned int> m_rowVelocity;
  tuple<unsigned int> m_columnVelocity;
  tuple<float> m_min;
  tuple<float> m_max;
  /// \brief velocity, stores to pointers to chunks of memory storing the velocities in x and y
  tuple<float *> m_velocity;
  tuple<float *> m_previousVelocity;
  tuple<float> * m_pvx;
  tuple<float> * m_pvy;
  void copy( tuple<float> * _src, tuple<float> * _dst, int _size );
  void copy( float * _src, float * _dst, int _size );


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
#endif // TESTING
};

#endif // _GPUSOLVER_H
